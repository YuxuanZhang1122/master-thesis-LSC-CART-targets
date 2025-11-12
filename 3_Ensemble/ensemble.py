import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    """Base class for all cell type predictors"""

    def __init__(self, name: str):
        self.name = name
        self.device = "mps"

    @abstractmethod
    def train(self, ref_data):
        """Train/fine-tune the model on reference data"""
        pass

    @abstractmethod
    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data"""
        pass

    def preprocess_data(self, adata, is_reference: bool = True):
        """Default preprocessing none - overridden by subclasses"""
        return adata.copy()

class CellTypeEnsemble:
    """Main ensemble class that contains all predictors"""

    def __init__(self, ref_path: str, query_path: str, n_hvg: int = 3000, status='infer'):
        self.ref_path = Path(ref_path)
        self.query_path = Path(query_path)
        self.n_hvg = n_hvg
        self.status = status
        self.predictors: Dict[str, BasePredictor] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.ref_data = None
        self.query_data = None
        self.consensus_predictions = None
        self.voting_breakdown = None

    def load_data(self):
        """Load reference and query data with HVG selection"""
        logger.info("Loading reference and query data...")
        self.ref_data = sc.read_h5ad(self.ref_path)
        self.query_data = sc.read_h5ad(self.query_path)

        logger.info(f"Reference data: {self.ref_data.n_obs} cells, {self.ref_data.n_vars} genes")
        logger.info(f"Query data: {self.query_data.n_obs} cells, {self.query_data.n_vars} genes")

        """
        Find HVGs with log-normalized reference dataset
        Subset both ref. and query to HVGs that are present in both
        """

        # Step 1: Select HVGs from full reference dataset
        logger.info(f"Selecting {self.n_hvg} highly variable genes from reference data...")
        adata_for_hvg = self.ref_data.copy()
        sc.pp.normalize_total(adata_for_hvg, target_sum=1e4)
        sc.pp.log1p(adata_for_hvg)

        # Use cell_ranger flavor
        sc.pp.highly_variable_genes(
            adata_for_hvg,
            n_top_genes=self.n_hvg,
            flavor='cell_ranger',
            batch_key='patient_id'
        )

        # Get HVG gene names
        hvg_mask = adata_for_hvg.var['highly_variable']
        hvg_genes = adata_for_hvg.var_names[hvg_mask]

        # Step 2: Find intersection of HVGs with query genes
        common_hvgs = hvg_genes.intersection(self.query_data.var_names)
        logger.info(f"Common HVGs between reference and query: {len(common_hvgs)}")

        # Step 3: Subset both datasets to final HVG intersection
        self.ref_data = self.ref_data[:, common_hvgs].copy()
        self.query_data = self.query_data[:, common_hvgs].copy()

    def add_predictor(self, predictor: BasePredictor):
        """Add a predictor to the ensemble"""
        self.predictors[predictor.name] = predictor
        logger.info(f"Added predictor: {predictor.name}")

    def run_predictions(self):
        """
        Preprocess -> train -> predict
        """

        for name, predictor in self.predictors.items():
            logger.info(f"Running {name}...")

            # Each predictor handles its own preprocessing
            ref_processed = predictor.preprocess_data(self.ref_data, is_reference=True)
            query_processed = predictor.preprocess_data(self.query_data, is_reference=False)

            # Train and predict
            predictor.train(ref_processed)
            predictions = predictor.predict(query_processed)

            self.predictions[name] = predictions

    def majority_vote(self):
        """Perform majority voting and generate breakdown"""

        n_cells = len(list(self.predictions.values())[0])
        consensus = []
        breakdown = []
        max_votes_list = []

        for cell_idx in range(n_cells):
            # Get all predictions for this cell
            cell_votes = {}
            cell_breakdown = {}

            for predictor_name, preds in self.predictions.items():
                vote = preds[cell_idx]
                cell_votes[vote] = cell_votes.get(vote, 0) + 1
                cell_breakdown[predictor_name] = vote

            # Find consensus (most voted label)
            consensus_label = max(cell_votes, key=cell_votes.get)
            max_votes = cell_votes[consensus_label]
            consensus.append(consensus_label)
            breakdown.append(cell_breakdown)
            max_votes_list.append(max_votes)

        self.consensus_predictions = np.array(consensus)
        self.voting_breakdown = breakdown
        self.max_votes = np.array(max_votes_list)

    def save_results(self, output_path: str = "ensemble_results.h5ad"):
        """Save all predictions and consensus to file"""

        # Create output AnnData object based on query
        result_adata = self.query_data.copy()

        # For evaluation, only work with labeled dataset
        if self.status == 'eval':
            result_adata.obs['true_labels'] = result_adata.obs['CellType_Merged'].copy()

        # Add consensus predictions
        result_adata.obs['consensus_prediction'] = self.consensus_predictions

        # Add high-confidence consensus labels (5+ and 6+ votes)
        consensus_5votes = np.where(self.max_votes >= 5, self.consensus_predictions, "uncertain")
        consensus_6votes = np.where(self.max_votes >= 6, self.consensus_predictions, "uncertain")
        result_adata.obs['consensus_label_5votes'] = consensus_5votes
        result_adata.obs['consensus_label_6votes'] = consensus_6votes

        # Add individual predictor results
        for predictor_name, preds in self.predictions.items():
            result_adata.obs[f'{predictor_name}_prediction'] = preds

        # Add voting breakdown as a string representation
        breakdown_strings = []
        for breakdown in self.voting_breakdown:
            breakdown_str = "; ".join([f"{pred}: {label}" for pred, label in breakdown.items()])
            breakdown_strings.append(breakdown_str)
        result_adata.obs['voting_breakdown'] = breakdown_strings

        # Save results
        output_path = Path(output_path)
        result_adata.write_h5ad(output_path)
        logger.info(f"Results saved to {output_path}")

        return result_adata

    def run_ensemble(self, output_path: str = "ensemble_results.h5ad"):
        """Run the complete ensemble pipeline

        Args:
            output_path: Path to save results
        """

        self.load_data() # HVG, subsetting
        self.run_predictions() # preprocess, train, and predict
        self.majority_vote()

        return self.save_results(output_path)


    def get_summary(self):
        """Get summary of ensemble results"""
        if self.consensus_predictions is None:
            return "No results available yet."

        summary = {
            'n_cells': len(self.consensus_predictions),
            'n_predictors': len(self.predictors),
            'consensus_cell_types': np.unique(self.consensus_predictions).tolist(),
            'individual_predictions': {name: np.unique(preds).tolist()
                                     for name, preds in self.predictions.items()}
        }
        return summary

if __name__ == "__main__":
    # Example usage
    #ensemble = CellTypeEnsemble("ref.h5ad", "query.h5ad")
    ensemble = CellTypeEnsemble("ref.h5ad", "query.h5ad")
    # Predictors will be added here
    # ensemble.add_predictor(ScANVIPredictor())
    # ensemble.add_predictor(ScPoliPredictor())
    # etc.

    # results = ensemble.run_ensemble()
    # print(ensemble.get_summary())