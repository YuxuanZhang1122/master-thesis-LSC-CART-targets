import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
import scanpy as sc
from pathlib import Path
import logging
from ensemble_analysis import EnsembleAnalyzer

logger = logging.getLogger(__name__)

class BinaryEnsembleAnalyzer(EnsembleAnalyzer):
    """Specialized analyzer for HSPC/LSPC binary classification ensemble results"""

    def __init__(self, results_path: str):
        super().__init__(results_path)
        logger.info("Initialized BinaryEnsembleAnalyzer for HSPC/LSPC analysis")

    def plot_combined_accuracy_agreement(self, figsize=(15, 6), save_path=None):

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Predictor Performance Overview', fontsize=14, fontweight='bold')

        # Left subplot: Individual accuracies
        ax1 = axes[0]
        accuracies = []
        for pred_name in self.predictor_names:
            accuracy = accuracy_score(self.true_labels, self.individual_predictions[pred_name])
            accuracies.append(accuracy)

        # Add consensus accuracy
        consensus_accuracy = accuracy_score(self.true_labels, self.consensus_labels)

        # Add high-confidence consensus accuracies (5+ and 6+ votes)
        consensus_5_labels = self.adata.obs['consensus_label_5votes'].values
        consensus_6_labels = self.adata.obs['consensus_label_6votes'].values

        mask_5 = consensus_5_labels != "uncertain"
        mask_6 = consensus_6_labels != "uncertain"

        consensus_5_accuracy = accuracy_score(self.true_labels[mask_5], consensus_5_labels[mask_5]) if mask_5.sum() > 0 else 0
        consensus_6_accuracy = accuracy_score(self.true_labels[mask_6], consensus_6_labels[mask_6]) if mask_6.sum() > 0 else 0

        all_labels = self.predictor_names + ['Consensus', 'Consensus_5', 'Consensus_6']
        all_accuracies = accuracies + [consensus_accuracy, consensus_5_accuracy, consensus_6_accuracy]

        bars = ax1.bar(all_labels, all_accuracies)

        # Color consensus bars differently
        bars[-3].set_color('gold')
        bars[-2].set_color('darkorange')
        bars[-1].set_color('darkgoldenrod')

        ax1.set_ylabel('Accuracy')
        ax1.set_title('Individual Predictor vs Consensus Accuracy')
        ax1.set_xticklabels(all_labels, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, all_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.08,
                    f'{acc:.3f}', ha='center', va='bottom')

        # Right subplot: Predictor agreement matrix
        ax2 = axes[1]
        n_predictors = len(self.predictor_names)
        agreement_matrix = np.zeros((n_predictors, n_predictors))

        for i, pred1 in enumerate(self.predictor_names):
            for j, pred2 in enumerate(self.predictor_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = (self.individual_predictions[pred1] == self.individual_predictions[pred2]).mean()
                    agreement_matrix[i, j] = agreement

        sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=self.predictor_names, yticklabels=self.predictor_names,
                   vmin=0, vmax=1, ax=ax2)
        ax2.set_title('Predictor Agreement Matrix')
        ax2.set_xlabel('Predictors')
        ax2.set_ylabel('Predictors')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return dict(zip(all_labels, all_accuracies)), agreement_matrix

    def analyze_combined_lspc_hspc_analysis(self, min_agreement_fraction=6/7, figsize=(18, 10), save_path=None):

        # Prepare voting scores if not available
        if self.voting_scores is None:
            self.calculate_voting_confidence()

        # Filter for LSPC and HSPC cells only
        lspc_hspc_mask = np.isin(self.true_labels, ['LSPC', 'HSPC'])
        lspc_hspc_true = self.true_labels[lspc_hspc_mask]
        lspc_hspc_consensus = self.consensus_labels[lspc_hspc_mask]

        # Get individual predictions for LSPC/HSPC
        lspc_hspc_individual = {}
        for pred_name in self.predictor_names:
            lspc_hspc_individual[pred_name] = self.individual_predictions[pred_name][lspc_hspc_mask]

        # Prepare high-confidence data
        n_predictors = len(self.predictor_names)
        min_votes = int(np.ceil(min_agreement_fraction * n_predictors))
        high_conf_mask = self.voting_scores['max_votes'] >= min_votes
        high_conf_indices = self.voting_scores[high_conf_mask]['cell_idx'].values

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'LSPC/HSPC Analysis and High-Confidence Performance (>= 6/7 Agreement)', fontsize=16, fontweight='bold')

        # Row 1, Plot 1: LSPC/HSPC Confusion Matrix
        ax1 = axes[0, 0]
        cm_lspc_hspc = confusion_matrix(lspc_hspc_true, lspc_hspc_consensus, labels=['HSPC', 'LSPC'])
        sns.heatmap(cm_lspc_hspc, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['HSPC', 'LSPC'], yticklabels=['HSPC', 'LSPC'], ax=ax1)
        ax1.set_title('LSPC/HSPC Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        # Calculate key metrics
        tn, fp, fn, tp = cm_lspc_hspc.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Row 1, Plot 2: Correct vs Incorrect Confidence
        ax2 = axes[0, 1]
        correct_conf = self.voting_scores[self.voting_scores['is_correct']]['confidence']
        incorrect_conf = self.voting_scores[~self.voting_scores['is_correct']]['confidence']

        # FlowJo-style smooth density curves
        from scipy.stats import gaussian_kde

        # Create smooth x-axis
        x_min = min(correct_conf.min(), incorrect_conf.min())
        x_max = max(correct_conf.max(), incorrect_conf.max())
        x = np.linspace(x_min, x_max, 200)

        # Calculate KDE for smooth curves
        if len(correct_conf) > 1:
            kde_correct = gaussian_kde(correct_conf)
            y_correct = kde_correct(x)
        else:
            y_correct = np.zeros_like(x)

        if len(incorrect_conf) > 1:
            kde_incorrect = gaussian_kde(incorrect_conf)
            y_incorrect = kde_incorrect(x)
        else:
            y_incorrect = np.zeros_like(x)

        # Plot smooth curves
        ax2.plot(x, y_correct, color='green', linewidth=2, label=f'Correct (n={len(correct_conf)})')
        ax2.fill_between(x, y_correct, alpha=0.3, color='green')

        ax2.plot(x, y_incorrect, color='red', linewidth=2, label=f'Incorrect (n={len(incorrect_conf)})')
        ax2.fill_between(x, y_incorrect, alpha=0.3, color='red')

        ax2.set_xlabel('Voting Confidence')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Row 1, Plot 3: Agreement threshold sensitivity analysis
        ax3 = axes[0, 2]
        thresholds = np.arange(0.5, 1.01, 0.05)
        subset_sizes = []
        subset_accuracies = []

        for thresh in thresholds:
            min_votes_thresh = int(np.ceil(thresh * n_predictors))
            thresh_mask = self.voting_scores['max_votes'] >= min_votes_thresh
            thresh_indices = self.voting_scores[thresh_mask]['cell_idx'].values

            if len(thresh_indices) > 0:
                thresh_true = self.true_labels[thresh_indices]
                thresh_consensus = self.consensus_labels[thresh_indices]
                thresh_accuracy = accuracy_score(thresh_true, thresh_consensus)
                subset_size = len(thresh_indices) / len(self.true_labels)
            else:
                thresh_accuracy = np.nan
                subset_size = 0

            subset_sizes.append(subset_size)
            subset_accuracies.append(thresh_accuracy)

        ax3_twin = ax3.twinx()

        ax3.plot(thresholds, subset_accuracies, 'b-o', label='Accuracy', markersize=4)
        ax3.set_xlabel('Agreement Threshold')
        ax3.set_ylabel('Accuracy', color='b')
        ax3.tick_params(axis='y', labelcolor='b')

        ax3_twin.plot(thresholds, subset_sizes, 'r-s', label='Subset Size', markersize=4)
        ax3_twin.set_ylabel('Subset Size (Fraction)', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')

        ax3.axvline(min_agreement_fraction, color='gray', linestyle='--', alpha=0.7)
        ax3.set_title('Threshold Sensitivity')
        ax3.grid(alpha=0.3)

        # Row 2, Plot 1: High-confidence LSPC/HSPC Confusion Matrix
        ax4 = axes[1, 0]
        if len(high_conf_indices) > 0:
            high_conf_true = self.true_labels[high_conf_indices]
            high_conf_consensus = self.consensus_labels[high_conf_indices]
            lspc_hspc_high_conf_mask = np.isin(high_conf_true, ['LSPC', 'HSPC'])

            if np.any(lspc_hspc_high_conf_mask):
                lspc_hspc_high_conf_true = high_conf_true[lspc_hspc_high_conf_mask]
                lspc_hspc_high_conf_consensus = high_conf_consensus[lspc_hspc_high_conf_mask]

                cm_lspc_hspc_high = confusion_matrix(lspc_hspc_high_conf_true, lspc_hspc_high_conf_consensus,
                                                   labels=['HSPC', 'LSPC'])

                sns.heatmap(cm_lspc_hspc_high, annot=True, fmt='d', cmap='Reds',
                           xticklabels=['HSPC', 'LSPC'], yticklabels=['HSPC', 'LSPC'], ax=ax4)
                ax4.set_title('High-Conf LSPC/HSPC\nConfusion Matrix')
                ax4.set_xlabel('Predicted')
                ax4.set_ylabel('True')

            else:
                ax4.text(0.5, 0.5, 'No LSPC/HSPC in\nhigh-confidence subset',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('High-Conf LSPC/HSPC')
        else:
            ax4.text(0.5, 0.5, 'No high-confidence\npredictions found',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('High-Conf LSPC/HSPC')

        # Row 2, Plot 2: LSPC/HSPC Recall Comparison
        ax5 = axes[1, 1]
        pred_lspc_recalls = []
        pred_hspc_recalls = []

        for pred_name in self.predictor_names:
            pred_all_predictions = self.individual_predictions[pred_name]

            # LSPC recall
            lspc_mask = self.true_labels == 'LSPC'
            if np.any(lspc_mask):
                lspc_recall = np.sum((self.true_labels == 'LSPC') & (pred_all_predictions == 'LSPC')) / np.sum(lspc_mask)
            else:
                lspc_recall = 0
            pred_lspc_recalls.append(lspc_recall)

            # HSPC recall
            hspc_mask = self.true_labels == 'HSPC'
            if np.any(hspc_mask):
                hspc_recall = np.sum((self.true_labels == 'HSPC') & (pred_all_predictions == 'HSPC')) / np.sum(hspc_mask)
            else:
                hspc_recall = 0
            pred_hspc_recalls.append(hspc_recall)

        # Add consensus recalls
        consensus_lspc_recall = np.sum((self.true_labels == 'LSPC') & (self.consensus_labels == 'LSPC')) / np.sum(self.true_labels == 'LSPC') if np.any(self.true_labels == 'LSPC') else 0
        consensus_hspc_recall = np.sum((self.true_labels == 'HSPC') & (self.consensus_labels == 'HSPC')) / np.sum(self.true_labels == 'HSPC') if np.any(self.true_labels == 'HSPC') else 0
        pred_lspc_recalls.append(consensus_lspc_recall)
        pred_hspc_recalls.append(consensus_hspc_recall)

        # Add high-confidence consensus recalls
        if len(high_conf_indices) > 0:
            high_conf_true = self.true_labels[high_conf_indices]
            high_conf_consensus = self.consensus_labels[high_conf_indices]

            high_conf_lspc_recall = np.sum((high_conf_true == 'LSPC') & (high_conf_consensus == 'LSPC')) / np.sum(high_conf_true == 'LSPC') if np.any(high_conf_true == 'LSPC') else 0
            high_conf_hspc_recall = np.sum((high_conf_true == 'HSPC') & (high_conf_consensus == 'HSPC')) / np.sum(high_conf_true == 'HSPC') if np.any(high_conf_true == 'HSPC') else 0
        else:
            high_conf_lspc_recall = 0
            high_conf_hspc_recall = 0

        pred_lspc_recalls.append(high_conf_lspc_recall)
        pred_hspc_recalls.append(high_conf_hspc_recall)

        x_pos = np.arange(len(self.predictor_names) + 2)
        width = 0.35

        # Create bars with improved y-axis limits for better visualization
        lspc_bars = ax5.bar(x_pos - width/2, pred_lspc_recalls, width, alpha=0.8, color='lightcoral')
        hspc_bars = ax5.bar(x_pos + width/2, pred_hspc_recalls, width, alpha=0.8, color='lightgreen')

        # Highlight best performers
        best_lspc_idx = np.argmax(pred_lspc_recalls)
        best_hspc_idx = np.argmax(pred_hspc_recalls)
        lspc_bars[best_lspc_idx].set_color('darkred')
        hspc_bars[best_hspc_idx].set_color('darkgreen')

        # So the legend won't block things
        min_recall = min(min(pred_lspc_recalls), min(pred_hspc_recalls))
        if min_recall > 0.75:
            location_recall = 'lower center'
        else:
            location_recall = 'upper left'

        ax5.set_xlabel('Predictors')
        ax5.set_ylabel('Recall')
        ax5.set_title('LSPC/HSPC Recall by Predictor')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(self.predictor_names + ['Consensus', 'High-Conf'], rotation=45, ha='right')
        ax5.legend(['LSPC', 'HSPC'], loc=location_recall, ncol=2)

        # Improved y-axis limits for better visualization of high values
        if min_recall > 0.75:
            ax5.set_ylim(0.5, 1.01)
        else:
            ax5.set_ylim(0, 1.0)
        ax5.grid(axis='y', alpha=0.3)

        # Row 2, Plot 3: LSPC/HSPC Precision Comparison
        ax6 = axes[1, 2]
        pred_lspc_precisions = []
        pred_hspc_precisions = []

        for pred_name in self.predictor_names:
            pred_all_predictions = self.individual_predictions[pred_name]

            # LSPC precision
            lspc_predicted_mask = pred_all_predictions == 'LSPC'
            if np.any(lspc_predicted_mask):
                lspc_precision = np.sum((self.true_labels == 'LSPC') & (pred_all_predictions == 'LSPC')) / np.sum(lspc_predicted_mask)
            else:
                lspc_precision = 0
            pred_lspc_precisions.append(lspc_precision)

            # HSPC precision
            hspc_predicted_mask = pred_all_predictions == 'HSPC'
            if np.any(hspc_predicted_mask):
                hspc_precision = np.sum((self.true_labels == 'HSPC') & (pred_all_predictions == 'HSPC')) / np.sum(hspc_predicted_mask)
            else:
                hspc_precision = 0
            pred_hspc_precisions.append(hspc_precision)

        # Add consensus precisions
        consensus_lspc_precision = np.sum((self.true_labels == 'LSPC') & (self.consensus_labels == 'LSPC')) / np.sum(self.consensus_labels == 'LSPC') if np.any(self.consensus_labels == 'LSPC') else 0
        consensus_hspc_precision = np.sum((self.true_labels == 'HSPC') & (self.consensus_labels == 'HSPC')) / np.sum(self.consensus_labels == 'HSPC') if np.any(self.consensus_labels == 'HSPC') else 0
        pred_lspc_precisions.append(consensus_lspc_precision)
        pred_hspc_precisions.append(consensus_hspc_precision)

        # Add high-confidence consensus precisions
        if len(high_conf_indices) > 0:
            high_conf_true = self.true_labels[high_conf_indices]
            high_conf_consensus = self.consensus_labels[high_conf_indices]

            high_conf_lspc_precision = np.sum((high_conf_true == 'LSPC') & (high_conf_consensus == 'LSPC')) / np.sum(high_conf_consensus == 'LSPC') if np.any(high_conf_consensus == 'LSPC') else 0
            high_conf_hspc_precision = np.sum((high_conf_true == 'HSPC') & (high_conf_consensus == 'HSPC')) / np.sum(high_conf_consensus == 'HSPC') if np.any(high_conf_consensus == 'HSPC') else 0
        else:
            high_conf_lspc_precision = 0
            high_conf_hspc_precision = 0

        pred_lspc_precisions.append(high_conf_lspc_precision)
        pred_hspc_precisions.append(high_conf_hspc_precision)

        # Create bars with improved y-axis limits for better visualization
        lspc_bars = ax6.bar(x_pos - width/2, pred_lspc_precisions, width, alpha=0.8, color='lightcoral')
        hspc_bars = ax6.bar(x_pos + width/2, pred_hspc_precisions, width, alpha=0.8, color='lightgreen')

        # Highlight best performers
        best_lspc_idx = np.argmax(pred_lspc_precisions)
        best_hspc_idx = np.argmax(pred_hspc_precisions)
        lspc_bars[best_lspc_idx].set_color('darkred')
        hspc_bars[best_hspc_idx].set_color('darkgreen')

        # So the legend won't block things
        min_precision = min(min(pred_lspc_precisions), min(pred_hspc_precisions))
        if min_recall > 0.75:
            location_precision = 'lower center'
        else:
            location_precision = 'upper left'

        ax6.set_xlabel('Predictors')
        ax6.set_ylabel('Precision')
        ax6.set_title('LSPC/HSPC Precision by Predictor')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(self.predictor_names + ['Consensus', 'High-Conf'], rotation=45, ha='right')
        ax6.legend(['LSPC', 'HSPC'], loc=location_precision, ncol=2)

        # Improved y-axis limits for better visualization of high values
        if min_precision > 0.75:
            ax6.set_ylim(0.5, 1.01)
        else:
            ax6.set_ylim(0, 1.0)
        ax6.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate cell counts for reporting
        lspc_count = np.sum(lspc_hspc_true == 'LSPC')
        hspc_count = np.sum(lspc_hspc_true == 'HSPC')

        if len(high_conf_indices) > 0:
            print(f"\nHigh-confidence subset: {len(high_conf_indices)}/{len(self.true_labels)} cells "
                  f"({len(high_conf_indices)/len(self.true_labels)*100:.1f}%)")
            overall_accuracy = accuracy_score(self.true_labels, self.consensus_labels)
            if len(high_conf_indices) > 0:
                high_conf_accuracy = accuracy_score(self.true_labels[high_conf_indices], self.consensus_labels[high_conf_indices])
                improvement = high_conf_accuracy - overall_accuracy
                print(f"High-confidence accuracy improvement: +{improvement:.3f}")

        return {
            'confusion_matrix': cm_lspc_hspc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'hspc_count': hspc_count,
            'lspc_count': lspc_count,
            'high_conf_size': len(high_conf_indices) if len(high_conf_indices) > 0 else 0
        }

    def generate_metrics_table(self, output_path):
        """Generate comprehensive metrics table for all models including high-confidence consensus"""
        metrics_list = []

        # Calculate metrics for individual predictors
        for pred_name in self.predictor_names:
            pred_labels = self.individual_predictions[pred_name]
            metrics = self._calculate_metrics(self.true_labels, pred_labels, pred_name)
            metrics_list.append(metrics)

        # Calculate metrics for consensus
        consensus_metrics = self._calculate_metrics(self.true_labels, self.consensus_labels, 'Consensus')
        metrics_list.append(consensus_metrics)

        # Calculate metrics for consensus_5 (5+ votes)
        consensus_5_labels = self.adata.obs['consensus_label_5votes'].values
        mask_5 = consensus_5_labels != "uncertain"
        if mask_5.sum() > 0:
            consensus_5_metrics = self._calculate_metrics(self.true_labels[mask_5], consensus_5_labels[mask_5], 'Consensus_5')
        else:
            consensus_5_metrics = {'Model': 'Consensus_5', 'Accuracy': 0, 'F1': 0, 'HSPC_Recall': 0, 'HSPC_Precision': 0, 'LSPC_Recall': 0, 'LSPC_Precision': 0}
        metrics_list.append(consensus_5_metrics)

        # Calculate metrics for consensus_6 (6+ votes)
        consensus_6_labels = self.adata.obs['consensus_label_6votes'].values
        mask_6 = consensus_6_labels != "uncertain"
        if mask_6.sum() > 0:
            consensus_6_metrics = self._calculate_metrics(self.true_labels[mask_6], consensus_6_labels[mask_6], 'Consensus_6')
        else:
            consensus_6_metrics = {'Model': 'Consensus_6', 'Accuracy': 0, 'F1': 0, 'HSPC_Recall': 0, 'HSPC_Precision': 0, 'LSPC_Recall': 0, 'LSPC_Precision': 0}
        metrics_list.append(consensus_6_metrics)

        df = pd.DataFrame(metrics_list)
        df.to_excel(output_path, index=False)

        return df

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate all metrics for a given prediction set"""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        hspc_recall = recall_score(y_true, y_pred, labels=['HSPC'], average='macro', zero_division=0)
        hspc_precision = precision_score(y_true, y_pred, labels=['HSPC'], average='macro', zero_division=0)
        lspc_recall = recall_score(y_true, y_pred, labels=['LSPC'], average='macro', zero_division=0)
        lspc_precision = precision_score(y_true, y_pred, labels=['LSPC'], average='macro', zero_division=0)

        return {
            'Model': model_name,
            'Accuracy': acc,
            'F1': f1,
            'HSPC_Recall': hspc_recall,
            'HSPC_Precision': hspc_precision,
            'LSPC_Recall': lspc_recall,
            'LSPC_Precision': lspc_precision
        }

    def generate_binary_report(self, output_dir="binary_ensemble_analysis", min_agreement_fraction=6/7):
        """Generate streamlined analysis report optimized for HSPC/LSPC binary classification"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Combined accuracy and agreement analysis
        print("\n1. PREDICTOR PERFORMANCE OVERVIEW")
        print("-" * 40)
        accuracies, agreement_matrix = self.plot_combined_accuracy_agreement(
            save_path=output_dir / "combined_accuracy_agreement.png"
        )

        # 2. Combined LSPC/HSPC and high-confidence analysis
        print("\n2. COMBINED LSPC/HSPC AND HIGH-CONFIDENCE ANALYSIS")
        print("-" * 55)
        combined_results = self.analyze_combined_lspc_hspc_analysis(
            min_agreement_fraction=min_agreement_fraction,
            save_path=output_dir / "combined_lspc_hspc_analysis.png"
        )

        # 3. Generate metrics table
        print("\n3. GENERATING METRICS TABLE")
        print("-" * 40)
        metrics_df = self.generate_metrics_table(output_dir / "metrics_table.xlsx")

        print(f"\nAll results saved to: {output_dir}")

        return {
            'accuracies': accuracies,
            'agreement_matrix': agreement_matrix,
            'combined_analysis': combined_results,
            'metrics_table': metrics_df
        }

if __name__ == "__main__":
    # Example usage
    path = 'internal_validation_genotyped'  # Update path as needed
    analyzer = BinaryEnsembleAnalyzer(f"{path}/ensemble_results.h5ad")
    results = analyzer.generate_binary_report(output_dir=f"{path}_binary_analysis")
    print("Binary analysis complete!")