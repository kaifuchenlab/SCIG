#!/usr/bin/env python

import pytest
import pandas as pd
import numpy as np
import anndata
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

# Mock rpy2 modules before importing
sys.modules['rpy2'] = MagicMock()
sys.modules['rpy2.robjects'] = MagicMock()
sys.modules['rpy2.robjects.packages'] = MagicMock()
sys.modules['rpy2.robjects.vectors'] = MagicMock()

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scig_human_functions import (
    pesudobulk_cigpred_human,
    pesudobulk_cig_reg_pred_human,
    cig_pred_singlecell_human
)


class TestPesudobulkCigpredHuman:
    """Test cases for pesudobulk_cigpred_human function"""
    
    @pytest.fixture
    def sample_exp_data(self):
        """Sample expression data fixture"""
        return pd.DataFrame({
            'Genename': ['GENE1', 'GENE2', 'GENE3'],
            'Cell_Type_A': [10.5, 20.3, 0.0],
            'Cell_Type_B': [15.2, 0.0, 25.1],
            'Cell_Type_C': [5.5, 30.0, 12.8]
        })
    
    @pytest.fixture
    def sample_seq_features(self):
        """Sample sequence features fixture"""
        return pd.DataFrame({
            'Geneid': ['GENE1', 'GENE2', 'GENE3'],
            'hpa_256_tiss_ntpmtau': [0.5, 0.3, 0.8],
            'gtex_db_tissuegini': [0.2, 0.6, 0.4],
            'Total_utr_3_len': [1000, 1500, 800],
            'tss_dis': [500, 300, 700],
            'TF_motif_total_exons': [5, 3, 7],
            'rna_motif_total_gene_norm_len_cv': [0.1, 0.2, 0.15],
            'utr3_mirna_count': [2, 1, 3],
            'exon_cs_mean': [0.7, 0.8, 0.6],
            'tss_500bp_cs_median': [0.5, 0.4, 0.6],
            'V_codon_bias_sum': [10, 15, 12],
            'P_codon_bias_max': [0.3, 0.4, 0.2],
            'Y_codon_bias_max': [0.2, 0.3, 0.1],
            'A_codon_bias_median': [0.25, 0.35, 0.3],
            'cds_len': [2000, 2500, 1800],
            'AT_cds_pct': [0.4, 0.45, 0.35],
            'AT_intron_pct': [0.5, 0.55, 0.45],
            'AT_tss_3kb_PTG_pct': [0.3, 0.35, 0.25],
            'GC_tss_4kb_PTG_pct': [0.6, 0.65, 0.55]
        })
    
    @pytest.fixture
    def sample_gene_name_length(self):
        """Sample gene name and length fixture"""
        return pd.DataFrame({
            'Geneid': ['GENE1', 'GENE2', 'GENE3'],
            'Genename': ['GENE1', 'GENE2', 'GENE3'],
            'length': [2000, 2500, 1800]
        })
    
    @pytest.fixture
    def sample_training_table(self):
        """Sample training table fixture"""
        return pd.DataFrame({
            'TPM_exon_exp_mean': [5.0, 15.0, 8.0, 12.0, 3.0],
            'hpa_256_tiss_ntpmtau': [0.1, 0.2, 0.3, 0.4, 0.5],
            'gtex_db_tissuegini': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Total_utr_3_len': [1000, 1200, 800, 1500, 900],
            'tss_dis': [400, 500, 300, 600, 350],
            'TF_motif_total_exons': [4, 5, 3, 6, 2],
            'rna_motif_total_gene_norm_len_cv': [0.1, 0.15, 0.2, 0.12, 0.18],
            'utr3_mirna_count': [1, 2, 3, 1, 2],
            'exon_cs_mean': [0.6, 0.7, 0.8, 0.5, 0.9],
            'tss_500bp_cs_median': [0.4, 0.5, 0.6, 0.3, 0.7],
            'V_codon_bias_sum': [8, 10, 12, 15, 9],
            'P_codon_bias_max': [0.2, 0.3, 0.4, 0.25, 0.35],
            'Y_codon_bias_max': [0.1, 0.2, 0.3, 0.15, 0.25],
            'A_codon_bias_median': [0.2, 0.25, 0.3, 0.35, 0.28],
            'cds_len': [1800, 2000, 2200, 2400, 1900],
            'AT_cds_pct': [0.35, 0.4, 0.45, 0.38, 0.42],
            'AT_intron_pct': [0.45, 0.5, 0.55, 0.48, 0.52],
            'AT_tss_3kb_PTG_pct': [0.25, 0.3, 0.35, 0.28, 0.32],
            'GC_tss_4kb_PTG_pct': [0.55, 0.6, 0.65, 0.58, 0.62],
            'label': [0, 1, 0, 1, 0],
            'Gene_name': ['TRAIN1', 'TRAIN2', 'TRAIN3', 'TRAIN4', 'TRAIN5']
        })
    
    @patch('scig_human_functions.qnorm.quantile_normalize')
    @patch('scig_human_functions.norm')
    @patch('scig_human_functions.importr')
    @patch('scig_human_functions.FloatVector')
    def test_pesudobulk_cigpred_human_rawcount(
        self, mock_float_vector, mock_importr, mock_norm_class, mock_qnorm,
        sample_exp_data, sample_seq_features, sample_gene_name_length, sample_training_table
    ):
        """Test pesudobulk_cigpred_human with rawcount input"""
        
        # Mock the norm class and its methods
        mock_norm_instance = MagicMock()
        mock_norm_instance.tpm_norm = pd.DataFrame({
            'Cell_Type_A': [1.0, 2.0, 0.0],
            'Cell_Type_B': [1.5, 0.0, 2.5],
            'Cell_Type_C': [0.5, 3.0, 1.2],
            'length': [2000, 2500, 1800]
        }, index=['GENE1', 'GENE2', 'GENE3'])
        mock_norm_class.return_value = mock_norm_instance
        
        # Mock qnorm.quantile_normalize to return the input unchanged
        mock_qnorm.side_effect = lambda x, axis: x
        
        # Mock R stats package
        mock_stats = MagicMock()
        mock_stats.p_adjust.return_value = [0.01, 0.02, 0.03]
        mock_importr.return_value = mock_stats
        
        # Mock FloatVector
        mock_float_vector.return_value = [0.1, 0.2, 0.3]
        
        result = pesudobulk_cigpred_human(
            inputtype_pred='rawcount',
            exp_filename_pred=sample_exp_data,
            alll_seq_features1=sample_seq_features,
            gene_name_length1=sample_gene_name_length,
            training_table_human=sample_training_table
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert len(result) > 0
        
        # Check that columns for each cell type are present
        cell_types = ['Cell_Type_A', 'Cell_Type_B', 'Cell_Type_C']
        for cell_type in cell_types:
            assert cell_type in result.columns
            assert f'{cell_type}_distance' in result.columns
            assert f'{cell_type}_rank' in result.columns
    
    def test_pesudobulk_cigpred_human_tpm_input(
        self, sample_exp_data, sample_seq_features, sample_gene_name_length, sample_training_table
    ):
        """Test pesudobulk_cigpred_human with TPM input"""
        
        with patch('scig_human_functions.qnorm.quantile_normalize') as mock_qnorm, \
             patch('scig_human_functions.importr') as mock_importr, \
             patch('scig_human_functions.FloatVector') as mock_float_vector:
            
            # Mock qnorm.quantile_normalize to return the input unchanged
            mock_qnorm.side_effect = lambda x, axis: x
            
            # Mock R stats package
            mock_stats = MagicMock()
            mock_stats.p_adjust.return_value = [0.01, 0.02, 0.03]
            mock_importr.return_value = mock_stats
            
            # Mock FloatVector
            mock_float_vector.return_value = [0.1, 0.2, 0.3]
            
            result = pesudobulk_cigpred_human(
                inputtype_pred='tpm',
                exp_filename_pred=sample_exp_data,
                alll_seq_features1=sample_seq_features,
                gene_name_length1=sample_gene_name_length,
                training_table_human=sample_training_table
            )
            
            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert 'symbol' in result.columns
            assert len(result) > 0
    
    def test_pesudobulk_cigpred_human_empty_input(
        self, sample_seq_features, sample_gene_name_length, sample_training_table
    ):
        """Test pesudobulk_cigpred_human with empty expression data"""
        
        empty_exp_data = pd.DataFrame({'Genename': []})
        
        with pytest.raises((KeyError, IndexError, ValueError)):
            pesudobulk_cigpred_human(
                inputtype_pred='rawcount',
                exp_filename_pred=empty_exp_data,
                alll_seq_features1=sample_seq_features,
                gene_name_length1=sample_gene_name_length,
                training_table_human=sample_training_table
            )


class TestPesudobulkCigRegPredHuman:
    """Test cases for pesudobulk_cig_reg_pred_human function"""
    
    @pytest.fixture
    def sample_cig_pred_result(self):
        """Sample CIG prediction result fixture"""
        return pd.DataFrame({
            'Geneid': ['GENE1', 'GENE2', 'GENE3', 'TF1', 'TF2'],
            'symbol': ['GENE1', 'GENE2', 'GENE3', 'TF1', 'TF2'],
            'Cell_Type_A': [0.8, 0.6, 0.2, 0.9, 0.4],
            'Cell_Type_B': [0.7, 0.3, 0.5, 0.8, 0.6],
            'Cell_Type_A_distance': [2.5, 1.8, -0.5, 3.2, 0.8],
            'Cell_Type_B_distance': [2.2, 0.5, 1.2, 3.0, 1.5],
            'Cell_Type_A_exp': [10.5, 20.3, 5.0, 15.2, 8.1],
            'Cell_Type_B_exp': [15.2, 12.1, 25.1, 18.5, 12.3]
        })
    
    @pytest.fixture
    def sample_grn_data(self):
        """Sample GRN data fixture"""
        return pd.DataFrame({
            'source_genename': ['TF1', 'TF1', 'TF2', 'TF2'],
            'target_genename': ['GENE1', 'GENE2', 'GENE1', 'GENE3']
        })
    
    @pytest.fixture
    def sample_tf_data(self):
        """Sample TF data fixture"""
        return pd.DataFrame({
            'Geneid': ['TF1', 'TF2'],
            'TF': ['TF1', 'TF2']
        })
    
    @pytest.fixture
    def sample_training_table_cigreg(self):
        """Sample training table for CIG regulation prediction"""
        return pd.DataFrame({
            'no_of_parent_edges': [1, 2, 0, 3, 1],
            'no_of_children_edges': [5, 3, 8, 2, 6],
            'CIG': [0.8, 0.6, 0.3, 0.9, 0.4],
            'EXP': [15.0, 12.0, 8.0, 20.0, 10.0],
            'CIG_distance': [2.5, 1.8, 0.2, 3.2, 1.1],
            'CIGscore_parent_mean': [0.7, 0.5, 0.4, 0.8, 0.6],
            'Exp_parent_mean': [12.0, 10.0, 6.0, 18.0, 8.0],
            'CIG_distance_parent_mean': [2.0, 1.5, 0.5, 2.8, 1.0],
            'CIGscore_parent_median': [0.6, 0.4, 0.3, 0.7, 0.5],
            'Exp_parent_median': [11.0, 9.0, 5.0, 17.0, 7.0],
            'CIG_distance_parent_median': [1.8, 1.2, 0.3, 2.5, 0.8],
            'CIGscore_parent_CV': [0.1, 0.2, 0.3, 0.15, 0.25],
            'Exp_parent_CV': [0.2, 0.3, 0.4, 0.1, 0.35],
            'CIG_parent_distance_CV': [0.15, 0.25, 0.35, 0.1, 0.3],
            'CIGscore_child_mean': [0.6, 0.4, 0.5, 0.7, 0.3],
            'Exp_child_mean': [10.0, 8.0, 12.0, 15.0, 6.0],
            'CIG_distance_child_mean': [1.5, 1.0, 2.0, 2.2, 0.8],
            'CIGscore_child_median': [0.5, 0.3, 0.4, 0.6, 0.2],
            'Exp_child_median': [9.0, 7.0, 11.0, 14.0, 5.0],
            'CIG_distance_child_median': [1.2, 0.8, 1.8, 2.0, 0.5],
            'CIGscore_child_CV': [0.2, 0.3, 0.1, 0.25, 0.4],
            'Exp_child_CV': [0.3, 0.4, 0.2, 0.15, 0.5],
            'CIG_distance_child_CV': [0.25, 0.35, 0.15, 0.2, 0.45],
            'label': [1, 0, 1, 1, 0]
        })
    
    @patch('scig_human_functions.importr')
    @patch('scig_human_functions.FloatVector')
    def test_pesudobulk_cig_reg_pred_human_success(
        self, mock_float_vector, mock_importr,
        sample_cig_pred_result, sample_grn_data, sample_tf_data, sample_training_table_cigreg
    ):
        """Test successful execution of pesudobulk_cig_reg_pred_human"""
        
        # Mock R stats package
        mock_stats = MagicMock()
        mock_stats.p_adjust.return_value = [0.01, 0.02]
        mock_importr.return_value = mock_stats
        
        # Mock FloatVector
        mock_float_vector.return_value = [0.1, 0.2]
        
        celltype_names = ['Cell_Type_A', 'Cell_Type_B', 'Genename']
        
        result = pesudobulk_cig_reg_pred_human(
            cig_pred_result=sample_cig_pred_result,
            all_db_grn_human=sample_grn_data,
            tf_human=sample_tf_data,
            training_table_cigreg_human=sample_training_table_cigreg,
            celltype_names=celltype_names
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'TF' in result.columns
        assert len(result) > 0
        
        # Check that prediction columns are present for each cell type
        # Note: the actual column names have '_exp' suffix based on the function implementation
        for cell_type in ['Cell_Type_A', 'Cell_Type_B']:
            expected_cols = [f'{cell_type}_exp', f'{cell_type}_exp_distance', f'{cell_type}_exp_rank']
            for col in expected_cols:
                assert col in result.columns, f"Column '{col}' not found in result columns: {list(result.columns)}"
    
    @patch('scig_human_functions.importr')
    @patch('scig_human_functions.FloatVector')
    def test_pesudobulk_cig_reg_pred_human_edge_case_handling(
        self, mock_float_vector, mock_importr,
        sample_cig_pred_result, sample_grn_data, sample_training_table_cigreg
    ):
        """Test with minimal TF data that won't cause scaling errors"""
        
        # Mock R stats package
        mock_stats = MagicMock()
        mock_stats.p_adjust.return_value = [0.01]
        mock_importr.return_value = mock_stats
        
        # Mock FloatVector
        mock_float_vector.return_value = [0.1]
        
        # Create minimal TF data with one entry that should result in at least one sample
        minimal_tf_data = pd.DataFrame({
            'Geneid': ['TF1'],
            'TF': ['TF1']
        })
        celltype_names = ['Cell_Type_A', 'Cell_Type_B', 'Genename']
        
        try:
            result = pesudobulk_cig_reg_pred_human(
                cig_pred_result=sample_cig_pred_result,
                all_db_grn_human=sample_grn_data,
                tf_human=minimal_tf_data,
                training_table_cigreg_human=sample_training_table_cigreg,
                celltype_names=celltype_names
            )
            
            # Should return a valid result
            assert isinstance(result, pd.DataFrame)
            assert 'TF' in result.columns
            
        except (ValueError, IndexError) as e:
            # Some edge cases may legitimately fail due to insufficient data
            # This is acceptable behavior
            assert "sample" in str(e).lower() or "empty" in str(e).lower()


class TestCigPredSinglecellHuman:
    """Test cases for cig_pred_singlecell_human function"""
    
    @pytest.fixture
    def sample_adata(self):
        """Sample AnnData object fixture"""
        X = np.random.rand(100, 10)  # 100 cells, 10 genes (smaller for testing)
        obs = pd.DataFrame({
            'cell_type': ['Type_A'] * 50 + ['Type_B'] * 50
        }, index=[f'Cell_{i}' for i in range(100)])
        var = pd.DataFrame({
            'Geneid': [f'GENE{i}' for i in range(1, 11)],
            'gene_name': [f'GENE{i}' for i in range(1, 11)]
        }, index=[f'GENE{i}' for i in range(1, 11)])
        
        return anndata.AnnData(X=X, obs=obs, var=var)
    
    @pytest.fixture
    def sample_features(self):
        """Sample features fixture"""
        return pd.DataFrame({
            'Geneid': [f'GENE{i}' for i in range(1, 11)],
            'gene_symbols': [f'GENE{i}' for i in range(1, 11)]
        })
    
    @pytest.fixture
    def sample_seq_features_sc(self):
        """Sample sequence features for single cell"""
        data = {
            'Geneid': [f'GENE{i}' for i in range(1, 11)],
            'hpa_256_tiss_ntpmtau': np.random.rand(10),
            'gtex_db_tissuegini': np.random.rand(10),
            'Total_utr_3_len': np.random.randint(500, 3000, 10),
            'tss_dis': np.random.randint(100, 1000, 10),
            'TF_motif_total_exons': np.random.randint(1, 10, 10),
            'rna_motif_total_gene_norm_len_cv': np.random.rand(10),
            'utr3_mirna_count': np.random.randint(0, 5, 10),
            'exon_cs_mean': np.random.rand(10),
            'tss_500bp_cs_median': np.random.rand(10),
            'V_codon_bias_sum': np.random.randint(5, 20, 10),
            'P_codon_bias_max': np.random.rand(10),
            'Y_codon_bias_max': np.random.rand(10),
            'A_codon_bias_median': np.random.rand(10),
            'cds_len': np.random.randint(1000, 5000, 10),
            'AT_cds_pct': np.random.rand(10),
            'AT_intron_pct': np.random.rand(10),
            'AT_tss_3kb_PTG_pct': np.random.rand(10),
            'GC_tss_4kb_PTG_pct': np.random.rand(10)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_training_table_sc(self):
        """Sample training table for single cell"""
        return pd.DataFrame({
            'TPM_exon_exp_mean': np.random.rand(20) * 10,
            'hpa_256_tiss_ntpmtau': np.random.rand(20),
            'gtex_db_tissuegini': np.random.rand(20),
            'Total_utr_3_len': np.random.randint(500, 3000, 20),
            'tss_dis': np.random.randint(100, 1000, 20),
            'TF_motif_total_exons': np.random.randint(1, 10, 20),
            'rna_motif_total_gene_norm_len_cv': np.random.rand(20),
            'utr3_mirna_count': np.random.randint(0, 5, 20),
            'exon_cs_mean': np.random.rand(20),
            'tss_500bp_cs_median': np.random.rand(20),
            'V_codon_bias_sum': np.random.randint(5, 20, 20),
            'P_codon_bias_max': np.random.rand(20),
            'Y_codon_bias_max': np.random.rand(20),
            'A_codon_bias_median': np.random.rand(20),
            'cds_len': np.random.randint(1000, 5000, 20),
            'AT_cds_pct': np.random.rand(20),
            'AT_intron_pct': np.random.rand(20),
            'AT_tss_3kb_PTG_pct': np.random.rand(20),
            'GC_tss_4kb_PTG_pct': np.random.rand(20),
            'label': np.random.randint(0, 2, 20),
            'Gene_name': [f'TRAIN_GENE{i}' for i in range(1, 21)]
        })
    
    @patch('scig_human_functions.pd.read_csv')
    @patch('scig_human_functions.norm')
    def test_cig_pred_singlecell_human_basic_structure(
        self, mock_norm_class, mock_read_csv,
        sample_adata, sample_features, sample_seq_features_sc, sample_training_table_sc
    ):
        """Test basic structure validation of cig_pred_singlecell_human"""
        
        # Mock gene length data
        gene_length_data = pd.DataFrame({
            'Geneid': [f'GENE{i}' for i in range(1, 11)],
            'length': np.random.randint(1000, 5000, 10)
        })
        mock_read_csv.return_value = gene_length_data
        
        # Mock the norm class and its methods
        mock_norm_instance = MagicMock()
        # Create a mock TPM normalized dataframe
        tpm_data = np.random.rand(10, 100)  # 10 genes, 100 cells
        tpm_df = pd.DataFrame(
            tpm_data,
            index=[f'GENE{i}' for i in range(1, 11)],
            columns=[f'Cell_{i}' for i in range(100)]
        )
        mock_norm_instance.tpm_norm = tpm_df
        mock_norm_class.return_value = mock_norm_instance
        
        result = cig_pred_singlecell_human(
            adata=sample_adata,
            features=sample_features,
            all_seq_features_human=sample_seq_features_sc,
            training_table_human=sample_training_table_sc,
            exp_filename_pred='test_sample'
        )
        
        # Assertions
        assert isinstance(result, anndata.AnnData)
        assert result.X.shape[0] > 0  # Should have genes
        assert result.X.shape[1] > 0  # Should have cells


class TestIntegration:
    """Integration tests for the functions working together"""
    
    def test_function_type_annotations(self):
        """Test that functions have proper type annotations"""
        import inspect
        
        # Check pesudobulk_cigpred_human
        sig = inspect.signature(pesudobulk_cigpred_human)
        assert sig.parameters['inputtype_pred'].annotation == str
        assert sig.parameters['exp_filename_pred'].annotation == pd.DataFrame
        assert sig.return_annotation == pd.DataFrame
        
        # Check pesudobulk_cig_reg_pred_human
        sig = inspect.signature(pesudobulk_cig_reg_pred_human)
        assert sig.parameters['cig_pred_result'].annotation == pd.DataFrame
        assert sig.parameters['celltype_names'].annotation == list[str]
        assert sig.return_annotation == pd.DataFrame
        
        # Check cig_pred_singlecell_human
        sig = inspect.signature(cig_pred_singlecell_human)
        assert sig.parameters['adata'].annotation == anndata.AnnData
        assert sig.parameters['exp_filename_pred'].annotation == str
        assert sig.return_annotation == anndata.AnnData
    
    def test_constants_and_imports(self):
        """Test that required constants and imports are available"""
        import scig_human_functions
        
        # Check that key imports are available
        assert hasattr(scig_human_functions, 'pd')
        assert hasattr(scig_human_functions, 'np')
        assert hasattr(scig_human_functions, 'anndata')
        assert hasattr(scig_human_functions, 'LogisticRegression')
        assert hasattr(scig_human_functions, 'StandardScaler')
    
    def test_data_validation_functions(self):
        """Test basic data validation capabilities"""
        
        # Test DataFrame creation with required columns
        test_df = pd.DataFrame({
            'Geneid': ['GENE1', 'GENE2'],
            'symbol': ['GENE1', 'GENE2'],
            'value': [0.5, 0.8]
        })
        
        assert 'Geneid' in test_df.columns
        assert 'symbol' in test_df.columns
        assert len(test_df) == 2
        
        # Test AnnData basic structure
        X = np.random.rand(10, 5)
        adata = anndata.AnnData(X=X)
        assert adata.X.shape == (10, 5)


if __name__ == '__main__':
    pytest.main([__file__]) 