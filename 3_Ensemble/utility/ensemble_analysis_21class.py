import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import scanpy as sc
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EnsembleAnalyzer:
    """Comprehensive analysis and visualization for ensemble results"""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.adata = None
        self.predictor_names = []
        self.true_labels = None
        self.consensus_labels = None
        self.individual_predictions = {}
        self.voting_scores = None

        self.load_results()

    def load_results(self):
        """Load ensemble results from h5ad file"""
        logger.info(f"Loading results from {self.results_path}")
        self.adata = sc.read_h5ad(self.results_path)

        # Extract labels
        self.true_labels = self.adata.obs['true_labels'].values
        self.consensus_labels = self.adata.obs['consensus_prediction'].values

        # Extract individual predictions
        for col in self.adata.obs.columns:
            if col.endswith('_prediction') and col != 'consensus_prediction':
                predictor_name = col.replace('_prediction', '')
                self.predictor_names.append(predictor_name)
                self.individual_predictions[predictor_name] = self.adata.obs[col].values

        logger.info(f"Loaded {len(self.predictor_names)} predictors: {self.predictor_names}")

    def calculate_voting_confidence(self):
        """Calculate voting confidence scores for each cell"""
        n_cells = len(self.true_labels)
        n_predictors = len(self.predictor_names)

        voting_scores = []

        for cell_idx in range(n_cells):
            # Get all votes for this cell
            votes = [self.individual_predictions[pred][cell_idx] for pred in self.predictor_names]
            vote_counts = Counter(votes)

            # Calculate confidence as max votes / total votes
            max_votes = max(vote_counts.values())
            confidence = max_votes / n_predictors

            voting_scores.append({
                'cell_idx': cell_idx,
                'confidence': confidence,
                'max_votes': max_votes,
                'total_votes': n_predictors,
                'consensus': self.consensus_labels[cell_idx],
                'true_label': self.true_labels[cell_idx],
                'is_correct': self.consensus_labels[cell_idx] == self.true_labels[cell_idx]
            })

        self.voting_scores = pd.DataFrame(voting_scores)
        return self.voting_scores

    def plot_confidence_by_celltype(self, figsize=(12, 8), save_path=None):
        """Plot mean voting confidence per cell type"""
        if self.voting_scores is None:
            self.calculate_voting_confidence()

        # Calculate mean confidence per cell type
        confidence_by_type = self.voting_scores.groupby('true_label')['confidence'].agg(['mean', 'std', 'count']).reset_index()
        confidence_by_type = confidence_by_type.sort_values('mean', ascending=False)

        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(confidence_by_type)), confidence_by_type['mean'],
                      yerr=confidence_by_type['std'], capsize=5, alpha=0.7)

        # Color bars by confidence level
        for i, bar in enumerate(bars):
            conf = confidence_by_type.iloc[i]['mean']
            if conf >= 0.8:
                bar.set_color('green')
            elif conf >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        plt.xlabel('Cell Types')
        plt.ylabel('Mean Voting Confidence')
        plt.title('Ensemble Confidence by Cell Type\n(Green: High ≥0.8, Orange: Medium ≥0.6, Red: Low <0.6)')
        plt.xticks(range(len(confidence_by_type)), confidence_by_type['true_label'], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return confidence_by_type

    def plot_correct_vs_incorrect_confidence(self, figsize=(8, 6), save_path=None):
        """Compare voting confidence for correctly vs incorrectly classified cells"""
        if self.voting_scores is None:
            self.calculate_voting_confidence()

        correct_conf = self.voting_scores[self.voting_scores['is_correct']]['confidence']
        incorrect_conf = self.voting_scores[~self.voting_scores['is_correct']]['confidence']

        plt.figure(figsize=figsize)

        # FlowJo-style smooth density curves
        from scipy.stats import gaussian_kde
        import numpy as np

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
        plt.plot(x, y_correct, color='green', linewidth=2, label=f'Correct (n={len(correct_conf)})')
        plt.fill_between(x, y_correct, alpha=0.3, color='green')

        plt.plot(x, y_incorrect, color='red', linewidth=2, label=f'Incorrect (n={len(incorrect_conf)})')
        plt.fill_between(x, y_incorrect, alpha=0.3, color='red')

        plt.xlabel('Voting Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics
        print(f"Correct predictions - Mean confidence: {correct_conf.mean():.3f} ± {correct_conf.std():.3f}")
        print(f"Incorrect predictions - Mean confidence: {incorrect_conf.mean():.3f} ± {incorrect_conf.std():.3f}")

    def plot_confusion_matrix(self, figsize=(10, 8), save_path=None):
        """Plot confusion matrix between true and consensus labels"""
        cm = confusion_matrix(self.true_labels, self.consensus_labels)

        # Get unique labels for ordering
        unique_labels = sorted(set(self.true_labels) | set(self.consensus_labels))

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted (Consensus)')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix: True vs Consensus Labels')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_predictor_agreement(self, figsize=(12, 8), save_path=None):
        """Plot agreement matrix between different predictors"""
        n_predictors = len(self.predictor_names)
        agreement_matrix = np.zeros((n_predictors, n_predictors))

        for i, pred1 in enumerate(self.predictor_names):
            for j, pred2 in enumerate(self.predictor_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = (self.individual_predictions[pred1] == self.individual_predictions[pred2]).mean()
                    agreement_matrix[i, j] = agreement

        plt.figure(figsize=figsize)
        sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=self.predictor_names, yticklabels=self.predictor_names,
                   vmin=0, vmax=1)
        plt.title('Predictor Agreement Matrix')
        plt.xlabel('Predictors')
        plt.ylabel('Predictors')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return agreement_matrix

    def plot_individual_accuracies(self, figsize=(10, 6), save_path=None):
        """Plot individual predictor accuracies"""
        accuracies = []
        for pred_name in self.predictor_names:
            accuracy = accuracy_score(self.true_labels, self.individual_predictions[pred_name])
            accuracies.append(accuracy)

        # Add consensus accuracy
        consensus_accuracy = accuracy_score(self.true_labels, self.consensus_labels)

        plt.figure(figsize=figsize)
        bars = plt.bar(self.predictor_names + ['Consensus'], accuracies + [consensus_accuracy])

        # Color consensus bar differently
        bars[-1].set_color('gold')

        plt.ylabel('Accuracy')
        plt.title('Individual Predictor vs Consensus Overall Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies + [consensus_accuracy])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.08,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return dict(zip(self.predictor_names + ['Consensus'], accuracies + [consensus_accuracy]))

    def analyze_lspc_hspc_performance(self, figsize=(15, 10), save_path=None):
        """Specialized analysis for LSPC (malignant) vs HSPC (normal) cell prediction"""

        # Filter for LSPC and HSPC cells only
        lspc_hspc_mask = np.isin(self.true_labels, ['LSPC', 'HSPC'])
        if not np.any(lspc_hspc_mask):
            print("Warning: No LSPC or HSPC cells found in the data.")
            return None

        lspc_hspc_true = self.true_labels[lspc_hspc_mask]
        lspc_hspc_consensus = self.consensus_labels[lspc_hspc_mask]

        # Get individual predictions for LSPC/HSPC
        lspc_hspc_individual = {}
        for pred_name in self.predictor_names:
            lspc_hspc_individual[pred_name] = self.individual_predictions[pred_name][lspc_hspc_mask]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('LSPC (Malignant) vs HSPC (Normal) Cell Analysis', fontsize=16, fontweight='bold')

        # 1. Confusion matrix for LSPC/HSPC only
        ax1 = axes[0, 0]
        cm_lspc_hspc = confusion_matrix(lspc_hspc_true, lspc_hspc_consensus, labels=['HSPC', 'LSPC'])
        sns.heatmap(cm_lspc_hspc, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['HSPC', 'LSPC'], yticklabels=['HSPC', 'LSPC'], ax=ax1)
        ax1.set_title('LSPC/HSPC Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        # Calculate key metrics
        tn, fp, fn, tp = cm_lspc_hspc.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # LSPC detection rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # HSPC correct classification rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # LSPC precision
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # 2. Misclassification analysis (moved from plot 6)
        ax2 = axes[0, 1]

        # Count misclassifications
        hspc_to_lspc = np.sum((lspc_hspc_true == 'HSPC') & (lspc_hspc_consensus == 'LSPC'))
        lspc_to_hspc = np.sum((lspc_hspc_true == 'LSPC') & (lspc_hspc_consensus == 'HSPC'))
        hspc_to_other = np.sum((lspc_hspc_true == 'HSPC') & (~np.isin(lspc_hspc_consensus, ['HSPC', 'LSPC'])))
        lspc_to_other = np.sum((lspc_hspc_true == 'LSPC') & (~np.isin(lspc_hspc_consensus, ['HSPC', 'LSPC'])))

        misclass_types = ['HSPC→LSPC\n(False Positive)', 'LSPC→HSPC\n(False Negative)',
                         'HSPC→Other', 'LSPC→Other']
        misclass_counts = [hspc_to_lspc, lspc_to_hspc, hspc_to_other, lspc_to_other]

        bars = ax2.bar(misclass_types, misclass_counts, color=['red', 'orange', 'yellow', 'pink'])
        ax2.set_title('Misclassification Patterns')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        # Add count labels
        for bar, count in zip(bars, misclass_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')

        # 3. Individual predictor LSPC/HSPC recall comparison (moved to bottom)
        ax3 = axes[1, 0]
        pred_lspc_recalls = []
        pred_hspc_recalls = []

        for pred_name in self.predictor_names:
            # Calculate recall for LSPC and HSPC using full dataset
            pred_all_predictions = self.individual_predictions[pred_name]

            # LSPC recall: correctly predicted LSPC / all true LSPC
            lspc_mask = self.true_labels == 'LSPC'
            if np.any(lspc_mask):
                lspc_recall = np.sum((self.true_labels == 'LSPC') & (pred_all_predictions == 'LSPC')) / np.sum(lspc_mask)
            else:
                lspc_recall = 0
            pred_lspc_recalls.append(lspc_recall)

            # HSPC recall: correctly predicted HSPC / all true HSPC
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

        # Calculate high-confidence subset recalls
        if self.voting_scores is None:
            self.calculate_voting_confidence()

        # Filter for high-confidence predictions (≥5/7 agreement)
        n_predictors = len(self.predictor_names)
        min_votes = int(np.ceil(5/7 * n_predictors))
        high_conf_mask = self.voting_scores['max_votes'] >= min_votes
        high_conf_indices = self.voting_scores[high_conf_mask]['cell_idx'].values

        if len(high_conf_indices) > 0:
            high_conf_consensus = self.consensus_labels[high_conf_indices]
            high_conf_true = self.true_labels[high_conf_indices]

            # High-confidence LSPC recall
            hc_lspc_mask = high_conf_true == 'LSPC'
            if np.any(hc_lspc_mask):
                hc_lspc_recall = np.sum((high_conf_true == 'LSPC') & (high_conf_consensus == 'LSPC')) / np.sum(hc_lspc_mask)
            else:
                hc_lspc_recall = 0

            # High-confidence HSPC recall
            hc_hspc_mask = high_conf_true == 'HSPC'
            if np.any(hc_hspc_mask):
                hc_hspc_recall = np.sum((high_conf_true == 'HSPC') & (high_conf_consensus == 'HSPC')) / np.sum(hc_hspc_mask)
            else:
                hc_hspc_recall = 0
        else:
            hc_lspc_recall = 0
            hc_hspc_recall = 0

        # Add high-confidence recalls
        pred_lspc_recalls.append(hc_lspc_recall)
        pred_hspc_recalls.append(hc_hspc_recall)

        x_pos = np.arange(len(self.predictor_names) + 2)  # +2 for consensus and high-confidence
        width = 0.35

        # Create bars with base colors
        lspc_bars = ax3.bar(x_pos - width/2, pred_lspc_recalls, width, alpha=0.8, color='lightcoral')
        hspc_bars = ax3.bar(x_pos + width/2, pred_hspc_recalls, width, alpha=0.8, color='lightgreen')

        # Highlight best performers
        best_lspc_idx = np.argmax(pred_lspc_recalls)
        best_hspc_idx = np.argmax(pred_hspc_recalls)
        lspc_bars[best_lspc_idx].set_color('darkred')
        hspc_bars[best_hspc_idx].set_color('darkgreen')

        ax3.set_xlabel('Predictors')
        ax3.set_ylabel('Recall')
        ax3.set_title('LSPC/HSPC Recall by Predictor')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.predictor_names + ['Consensus', 'Confident\nConsensus'], rotation=45, ha='right')
        ax3.legend(['LSPC', 'HSPC'], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        ax3.set_ylim(0, 1)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Individual predictor LSPC/HSPC precision comparison
        ax4 = axes[1, 1]
        pred_lspc_precisions = []
        pred_hspc_precisions = []

        for pred_name in self.predictor_names:
            # Calculate precision for LSPC and HSPC using full dataset
            pred_all_predictions = self.individual_predictions[pred_name]

            # LSPC precision: correctly predicted LSPC / all predicted LSPC
            lspc_predicted_mask = pred_all_predictions == 'LSPC'
            if np.any(lspc_predicted_mask):
                lspc_precision = np.sum((self.true_labels == 'LSPC') & (pred_all_predictions == 'LSPC')) / np.sum(lspc_predicted_mask)
            else:
                lspc_precision = 0
            pred_lspc_precisions.append(lspc_precision)

            # HSPC precision: correctly predicted HSPC / all predicted HSPC
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

        # Calculate high-confidence subset precisions (using the same high_conf data from recall section)
        if len(high_conf_indices) > 0 and 'high_conf_consensus' in locals() and 'high_conf_true' in locals():
            # High-confidence LSPC precision
            hc_lspc_predicted_mask = high_conf_consensus == 'LSPC'
            if np.any(hc_lspc_predicted_mask):
                hc_lspc_precision = np.sum((high_conf_true == 'LSPC') & (high_conf_consensus == 'LSPC')) / np.sum(hc_lspc_predicted_mask)
            else:
                hc_lspc_precision = 0

            # High-confidence HSPC precision
            hc_hspc_predicted_mask = high_conf_consensus == 'HSPC'
            if np.any(hc_hspc_predicted_mask):
                hc_hspc_precision = np.sum((high_conf_true == 'HSPC') & (high_conf_consensus == 'HSPC')) / np.sum(hc_hspc_predicted_mask)
            else:
                hc_hspc_precision = 0
        else:
            hc_lspc_precision = 0
            hc_hspc_precision = 0

        # Add high-confidence precisions
        pred_lspc_precisions.append(hc_lspc_precision)
        pred_hspc_precisions.append(hc_hspc_precision)

        x_pos = np.arange(len(self.predictor_names) + 2)  # +2 for consensus and high-confidence
        width = 0.35

        # Create bars with base colors
        lspc_bars = ax4.bar(x_pos - width/2, pred_lspc_precisions, width, alpha=0.8, color='lightcoral')
        hspc_bars = ax4.bar(x_pos + width/2, pred_hspc_precisions, width, alpha=0.8, color='lightgreen')

        # Highlight best performers
        best_lspc_idx = np.argmax(pred_lspc_precisions)
        best_hspc_idx = np.argmax(pred_hspc_precisions)
        lspc_bars[best_lspc_idx].set_color('darkred')
        hspc_bars[best_hspc_idx].set_color('darkgreen')

        ax4.set_xlabel('Predictors')
        ax4.set_ylabel('Precision')
        ax4.set_title('LSPC/HSPC Precision by Predictor')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(self.predictor_names + ['Consensus', 'Confident\nConsensus'], rotation=45, ha='right')
        ax4.legend(['LSPC', 'HSPC'], loc='upper left', bbox_to_anchor=(0, 1), ncol=2)
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Calculate cell counts for reporting
        lspc_count = np.sum(lspc_hspc_true == 'LSPC')
        hspc_count = np.sum(lspc_hspc_true == 'HSPC')

        # Print detailed report
        print("\n" + "="*60)
        print("LSPC vs HSPC DETAILED ANALYSIS")
        print("="*60)
        print(f"Total HSPC cells: {hspc_count}")
        print(f"Total LSPC cells: {lspc_count}")
        print(f"LSPC detection sensitivity: {sensitivity:.3f}")
        print(f"HSPC classification specificity: {specificity:.3f}")
        print(f"LSPC precision: {precision:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"False positives (HSPC→LSPC): {hspc_to_lspc}")
        print(f"False negatives (LSPC→HSPC): {lspc_to_hspc}")

        return {
            'confusion_matrix': cm_lspc_hspc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'false_positives': hspc_to_lspc,
            'false_negatives': lspc_to_hspc,
            'hspc_count': hspc_count,
            'lspc_count': lspc_count
        }

    def analyze_high_confidence_subset(self, min_agreement_fraction=5/7, figsize=(15, 8), save_path=None):
        """Analyze performance on high-confidence predictions (e.g., 5/7 predictors agree)"""

        if self.voting_scores is None:
            self.calculate_voting_confidence()

        # Convert fraction to minimum votes needed
        n_predictors = len(self.predictor_names)
        min_votes = int(np.ceil(min_agreement_fraction * n_predictors))

        # Filter high-confidence predictions
        high_conf_mask = self.voting_scores['max_votes'] >= min_votes
        high_conf_indices = self.voting_scores[high_conf_mask]['cell_idx'].values

        if len(high_conf_indices) == 0:
            print(f"No cells found with ≥{min_votes}/{n_predictors} predictor agreement")
            return None

        high_conf_true = self.true_labels[high_conf_indices]
        high_conf_consensus = self.consensus_labels[high_conf_indices]

        print(f"High-confidence subset: {len(high_conf_indices)}/{len(self.true_labels)} cells "
              f"({len(high_conf_indices)/len(self.true_labels)*100:.1f}%)")
        print(f"Agreement threshold: ≥{min_votes}/{n_predictors} predictors ({min_agreement_fraction:.1%})")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'High-Confidence Predictions Analysis (≥{min_votes}/{n_predictors} Agreement)',
                    fontsize=14, fontweight='bold')

        # 1. Confusion matrix for high-confidence subset
        ax1 = axes[0, 0]
        unique_labels = sorted(set(high_conf_true) | set(high_conf_consensus))
        cm_high_conf = confusion_matrix(high_conf_true, high_conf_consensus, labels=unique_labels)

        sns.heatmap(cm_high_conf, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_labels, yticklabels=unique_labels, ax=ax1)
        ax1.set_title('High-Confidence Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=0)

        # 2. Accuracy comparison: All vs High-confidence
        ax2 = axes[0, 1]
        overall_accuracy = accuracy_score(self.true_labels, self.consensus_labels)
        high_conf_accuracy = accuracy_score(high_conf_true, high_conf_consensus)

        bars = ax2.bar(['All Predictions', 'High-Confidence\nSubset'],
                      [overall_accuracy, high_conf_accuracy],
                      color=['lightblue', 'darkgreen'])
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Overall Accuracy Improvement')
        ax2.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, [overall_accuracy, high_conf_accuracy]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.08,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Calculate improvement
        improvement = high_conf_accuracy - overall_accuracy
        ax2.text(0.5, 0.7, f'Improvement: +{improvement:.3f}',
                transform=ax2.transAxes, ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 3. Cell type distribution in high-confidence subset
        ax3 = axes[0, 2]
        high_conf_type_counts = pd.Series(high_conf_true).value_counts()
        all_type_counts = pd.Series(self.true_labels).value_counts()

        # Calculate retention rate per cell type
        retention_rates = {}
        for cell_type in all_type_counts.index:
            if cell_type in high_conf_type_counts.index:
                retention_rates[cell_type] = high_conf_type_counts[cell_type] / all_type_counts[cell_type]
            else:
                retention_rates[cell_type] = 0

        sorted_types = sorted(retention_rates.keys(), key=lambda x: retention_rates[x], reverse=True)
        retention_values = [retention_rates[ct] for ct in sorted_types]

        bars = ax3.bar(range(len(sorted_types)), retention_values)
        ax3.set_xlabel('Cell Types')
        ax3.set_ylabel('Retention Rate')
        ax3.set_title('High-Confidence Retention\nby Cell Type')
        ax3.set_xticks(range(len(sorted_types)))
        ax3.set_xticklabels(sorted_types, rotation=45, ha='right')
        ax3.set_ylim(0, 1)

        # Color bars by retention rate
        for bar, rate in zip(bars, retention_values):
            if rate >= 0.8:
                bar.set_color('green')
            elif rate >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # 4. LSPC/HSPC performance in high-confidence subset
        ax4 = axes[1, 0]
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

            # Calculate metrics for high-confidence LSPC/HSPC
            if cm_lspc_hspc_high.shape == (2, 2):
                tn, fp, fn, tp = cm_lspc_hspc_high.ravel()
                hc_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                hc_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                hc_sensitivity, hc_specificity = 0, 0
        else:
            ax4.text(0.5, 0.5, 'No LSPC/HSPC in\nhigh-confidence subset',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('High-Conf LSPC/HSPC')
            hc_sensitivity, hc_specificity = 0, 0

        # 5. Confidence score distribution comparison
        ax5 = axes[1, 1]
        all_confidence = self.voting_scores['confidence']
        high_conf_confidence = self.voting_scores[high_conf_mask]['confidence']

        ax5.hist(all_confidence, bins=20, alpha=0.7, label='All Predictions', density=True, color='lightblue')
        ax5.hist(high_conf_confidence, bins=20, alpha=0.7, label='High-Confidence', density=True, color='darkgreen')
        ax5.axvline(min_agreement_fraction, color='red', linestyle='--', label=f'Threshold ({min_agreement_fraction:.1%})')
        ax5.set_xlabel('Voting Confidence')
        ax5.set_ylabel('Density')
        ax5.set_title('Confidence Distribution')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Agreement threshold sensitivity analysis
        ax6 = axes[1, 2]
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

        ax6_twin = ax6.twinx()

        line1 = ax6.plot(thresholds, subset_accuracies, 'b-o', label='Accuracy', markersize=4)
        ax6.set_xlabel('Agreement Threshold')
        ax6.set_ylabel('Accuracy', color='b')
        ax6.tick_params(axis='y', labelcolor='b')

        line2 = ax6_twin.plot(thresholds, subset_sizes, 'r-s', label='Subset Size', markersize=4)
        ax6_twin.set_ylabel('Subset Size (Fraction)', color='r')
        ax6_twin.tick_params(axis='y', labelcolor='r')

        ax6.axvline(min_agreement_fraction, color='gray', linestyle='--', alpha=0.7)
        ax6.set_title('Threshold Sensitivity')
        ax6.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary
        print(f"\nHigh-confidence subset accuracy: {high_conf_accuracy:.3f}")
        print(f"Overall accuracy: {overall_accuracy:.3f}")
        print(f"Accuracy improvement: +{improvement:.3f}")

        if np.any(lspc_hspc_high_conf_mask):
            print(f"High-confidence LSPC sensitivity: {hc_sensitivity:.3f}")
            print(f"High-confidence HSPC specificity: {hc_specificity:.3f}")

        return {
            'high_conf_accuracy': high_conf_accuracy,
            'overall_accuracy': overall_accuracy,
            'improvement': improvement,
            'subset_size': len(high_conf_indices),
            'subset_fraction': len(high_conf_indices) / len(self.true_labels),
            'confusion_matrix': cm_high_conf,
            'lspc_sensitivity': hc_sensitivity,
            'hspc_specificity': hc_specificity,
            'retention_rates': retention_rates
        }

    def generate_full_report(self, output_dir="ensemble_analysis_results", include_lspc_hspc=True,
                           include_high_confidence=True, min_agreement_fraction=4/6):
        """Generate comprehensive analysis report with all visualizations including specialized analyses"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info("Generating comprehensive ensemble analysis report...")

        # 1. Basic ensemble analysis
        print("="*60)
        print("ENSEMBLE ANALYSIS REPORT")
        print("="*60)

        confidence_stats = self.plot_confidence_by_celltype(
            save_path=output_dir / "confidence_by_celltype.png"
        )

        # 2. Correct vs incorrect analysis
        self.plot_correct_vs_incorrect_confidence(
            save_path=output_dir / "correct_vs_incorrect_confidence.png"
        )

        # 3. Confusion matrix
        cm = self.plot_confusion_matrix(
            save_path=output_dir / "confusion_matrix.png"
        )

        # 4. Predictor agreement
        agreement_matrix = self.plot_predictor_agreement(
            save_path=output_dir / "predictor_agreement.png"
        )

        # 5. Individual accuracies
        accuracies = self.plot_individual_accuracies(
            save_path=output_dir / "individual_accuracies.png"
        )

        # 6. LSPC/HSPC specialized analysis
        lspc_hspc_results = None
        if include_lspc_hspc:
            print("\n" + "="*60)
            print("LSPC/HSPC SPECIALIZED ANALYSIS")
            print("="*60)

            lspc_hspc_results = self.analyze_lspc_hspc_performance(
                save_path=output_dir / "lspc_hspc_analysis.png"
            )

        # 7. High-confidence subset analysis
        high_conf_results = None
        if include_high_confidence:
            print("\n" + "="*60)
            print("HIGH-CONFIDENCE SUBSET ANALYSIS")
            print("="*60)

            high_conf_results = self.analyze_high_confidence_subset(
                min_agreement_fraction=min_agreement_fraction,
                save_path=output_dir / "high_confidence_analysis.png"
            )

        # 8. Generate comprehensive summary statistics
        self.save_comprehensive_summary_stats(output_dir, confidence_stats, accuracies,
                                            lspc_hspc_results, high_conf_results)

        print(f"\nAll results saved to: {output_dir}")

        return {
            'confidence_stats': confidence_stats,
            'accuracies': accuracies,
            'agreement_matrix': agreement_matrix,
            'confusion_matrix': cm,
            'lspc_hspc_results': lspc_hspc_results,
            'high_confidence_results': high_conf_results
        }

    def save_summary_stats(self, output_dir, confidence_stats, accuracies):
        """Save summary statistics to text file"""
        summary_path = output_dir / "summary_report.txt"

        with open(summary_path, 'w') as f:
            f.write("ENSEMBLE ANALYSIS SUMMARY REPORT\n")
            f.write("="*50 + "\n\n")

            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*20 + "\n")
            for pred, acc in accuracies.items():
                f.write(f"{pred}: {acc:.3f}\n")
            f.write("\n")

            f.write("CONFIDENCE STATISTICS:\n")
            f.write("-"*22 + "\n")
            if self.voting_scores is not None:
                overall_conf = self.voting_scores['confidence'].mean()
                f.write(f"Overall mean confidence: {overall_conf:.3f}\n")

                correct_conf = self.voting_scores[self.voting_scores['is_correct']]['confidence'].mean()
                incorrect_conf = self.voting_scores[~self.voting_scores['is_correct']]['confidence'].mean()
                f.write(f"Correct predictions confidence: {correct_conf:.3f}\n")
                f.write(f"Incorrect predictions confidence: {incorrect_conf:.3f}\n\n")

            f.write("TOP 5 MOST CONFIDENT CELL TYPES:\n")
            f.write("-"*35 + "\n")
            for _, row in confidence_stats.head().iterrows():
                f.write(f"{row['true_label']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})\n")

            f.write("\nBOTTOM 5 LEAST CONFIDENT CELL TYPES:\n")
            f.write("-"*37 + "\n")
            for _, row in confidence_stats.tail().iterrows():
                f.write(f"{row['true_label']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})\n")

        logger.info(f"Summary statistics saved to {summary_path}")

    def save_comprehensive_summary_stats(self, output_dir, confidence_stats, accuracies,
                                       lspc_hspc_results=None, high_conf_results=None):
        """Save comprehensive summary statistics including specialized analyses"""
        summary_path = output_dir / "comprehensive_summary_report.txt"

        with open(summary_path, 'w') as f:
            f.write("COMPREHENSIVE ENSEMBLE ANALYSIS SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")

            # Basic performance
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*20 + "\n")
            for pred, acc in accuracies.items():
                f.write(f"{pred}: {acc:.3f}\n")
            f.write("\n")

            # Confidence statistics
            f.write("CONFIDENCE STATISTICS:\n")
            f.write("-"*22 + "\n")
            if self.voting_scores is not None:
                overall_conf = self.voting_scores['confidence'].mean()
                f.write(f"Overall mean confidence: {overall_conf:.3f}\n")

                correct_conf = self.voting_scores[self.voting_scores['is_correct']]['confidence'].mean()
                incorrect_conf = self.voting_scores[~self.voting_scores['is_correct']]['confidence'].mean()
                f.write(f"Correct predictions confidence: {correct_conf:.3f}\n")
                f.write(f"Incorrect predictions confidence: {incorrect_conf:.3f}\n\n")

            # LSPC/HSPC analysis results
            if lspc_hspc_results is not None:
                f.write("LSPC (MALIGNANT) vs HSPC (NORMAL) ANALYSIS:\n")
                f.write("-"*45 + "\n")
                f.write(f"HSPC cells analyzed: {lspc_hspc_results['hspc_count']}\n")
                f.write(f"LSPC cells analyzed: {lspc_hspc_results['lspc_count']}\n")
                f.write(f"LSPC detection sensitivity: {lspc_hspc_results['sensitivity']:.3f}\n")
                f.write(f"HSPC classification specificity: {lspc_hspc_results['specificity']:.3f}\n")
                f.write(f"LSPC precision: {lspc_hspc_results['precision']:.3f}\n")
                f.write(f"F1-score: {lspc_hspc_results['f1_score']:.3f}\n")
                f.write(f"False positives (HSPC→LSPC): {lspc_hspc_results['false_positives']}\n")
                f.write(f"False negatives (LSPC→HSPC): {lspc_hspc_results['false_negatives']}\n\n")

            # High-confidence analysis results
            if high_conf_results is not None:
                f.write("HIGH-CONFIDENCE SUBSET ANALYSIS:\n")
                f.write("-"*35 + "\n")
                f.write(f"High-confidence subset size: {high_conf_results['subset_size']} cells "
                       f"({high_conf_results['subset_fraction']:.1%})\n")
                f.write(f"Overall accuracy: {high_conf_results['overall_accuracy']:.3f}\n")
                f.write(f"High-confidence accuracy: {high_conf_results['high_conf_accuracy']:.3f}\n")
                f.write(f"Accuracy improvement: +{high_conf_results['improvement']:.3f}\n")
                f.write(f"High-confidence LSPC sensitivity: {high_conf_results['lspc_sensitivity']:.3f}\n")
                f.write(f"High-confidence HSPC specificity: {high_conf_results['hspc_specificity']:.3f}\n\n")

                f.write("RETENTION RATES BY CELL TYPE:\n")
                f.write("-"*30 + "\n")
                for cell_type, rate in sorted(high_conf_results['retention_rates'].items(),
                                            key=lambda x: x[1], reverse=True):
                    f.write(f"{cell_type}: {rate:.3f}\n")
                f.write("\n")

            # Most and least confident cell types
            f.write("TOP 5 MOST CONFIDENT CELL TYPES:\n")
            f.write("-"*35 + "\n")
            for _, row in confidence_stats.head().iterrows():
                f.write(f"{row['true_label']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})\n")

            f.write("\nBOTTOM 5 LEAST CONFIDENT CELL TYPES:\n")
            f.write("-"*37 + "\n")
            for _, row in confidence_stats.tail().iterrows():
                f.write(f"{row['true_label']}: {row['mean']:.3f} ± {row['std']:.3f} (n={row['count']})\n")

        logger.info(f"Comprehensive summary statistics saved to {summary_path}")

if __name__ == "__main__":
    # Example usage
    path = 'ensemble_analysis_results'
    analyzer = EnsembleAnalyzer(f"{path}/ensemble_results.h5ad")
    results = analyzer.generate_full_report(output_dir=path)
    print("Analysis complete!")