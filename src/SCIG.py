#!/usr/bin/env python

import sys
import argparse
import pandas as pd
import os
import numpy as np
import anndata
import scanpy as sc
from scipy import io
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from scig_human_functions import *
from scig_mouse_functions import *


path = Path.cwd()
#print (path)
os.chdir(path)
def printHelp() -> None:
    
    help_text = """
Error found in your command. See the README for more information on using SCIG.
"""
    print(help_text)

def main() -> None:
    # parse arguments
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        usage="...",description='', 
        epilog="Chen lab, Boston Children's Hospital/Harvard Medical School", 
        formatter_class=CustomFormatter
    )
    parser.add_argument(
        '-organism', dest='organism', type=str, 
        help='Mention the organism name of a sample/data'
    )
    parser.add_argument(
        '-assaytype', dest='assaytype', type=str, 
        help='Type of the data either bulk/pseudobulk or single-cell RNA expression data'
    )
    parser.add_argument(
        '-inputtype', dest='inputtype', type=str,
        help='Mention whether the expression values are in TPM or in raw count'
    )
    parser.add_argument(
        '-file', dest='file', type=str, help='Mention a filename that contains the expression values of genes'
    )
    parser.add_argument(
        '-datadir', dest='datadir', type=str, default='../data',
        help='Path to the data directory containing required reference files'
    )
    args = parser.parse_args()
    if args is not None:
        organism_name_pred = args.organism
        assaytype_pred = args.assaytype
        inputtype_pred = args.inputtype
        exp_filename_pred = args.file
        data_dir = Path(args.datadir)
    else:
        printHelp()
        sys.exit(1)
        
    ##add sequence features and model training table
    if organism_name_pred =='hs':
        all_seq_features_human = pd.read_csv(data_dir / "Requried_seq_features_cigpred_human.txt", sep="\t")
        gene_name_length_human = pd.read_csv(data_dir / "gene_name_map_with_length_human.txt", sep="\t")
        training_table_human = pd.read_csv(data_dir / "training_table_human.txt", sep="\t")   
        training_table_cigreg_human = pd.read_csv(data_dir / "training_table_master_tf_cigs_human.txt", sep="\t")
        all_db_grn_human = pd.read_csv(data_dir / "GRN_network_compiled_human.txt", sep="\t")
        tf_human = pd.read_csv(data_dir / "tf_human_list.txt", sep="\t")
    elif organism_name_pred =='mm':
        all_seq_features_mouse = pd.read_csv(data_dir / "Requried_seq_features_cigpred_mouse.txt", sep="\t")
        gene_name_length_mouse = pd.read_csv(data_dir / "gene_name_map_with_length_mouse.txt", sep="\t")
        #gene_name_length_mouse['Genename'] = gene_name_length_mouse['Genename'].str.upper()
        training_table_mouse = pd.read_csv(data_dir / "training_table_mouse.txt", sep="\t")
        training_table_cigreg_mouse = pd.read_csv(data_dir / "training_table_master_tf_cigs_mouse.txt", sep="\t")
        all_db_grn_mouse = pd.read_csv(data_dir / "all_db_GRN_combined_mouse.txt", sep="\t")
        tf_mouse = pd.read_csv(data_dir / "tf_mouse_list_geneid.txt", sep="\t")

    
    if organism_name_pred == 'hs' and assaytype_pred == 'bulk' and exp_filename_pred is not None:
        print ("HUMAN",organism_name_pred,assaytype_pred,inputtype_pred,exp_filename_pred)
        exp_matrix = pd.read_csv(exp_filename_pred, sep="\t")
        celltype_names = exp_matrix.columns.to_list()
        #exp_matrix.rename(columns = {str(celltype_names[0]): 'Genename'}, inplace = True)
        #print(exp_matrix)
        cig_pred_output_table = pesudobulk_cigpred_human(inputtype_pred, exp_matrix, all_seq_features_human, gene_name_length_human, training_table_human)
        first_column_symbol = cig_pred_output_table.pop('symbol')
        cig_pred_output_table.insert(0, 'symbol', first_column_symbol)
        output_file = Path(exp_filename_pred).with_suffix('.out').with_name(Path(exp_filename_pred).stem + 'cig_pred_result.out')
        cig_pred_output_table.to_csv(output_file, index=False, sep="\t")
        
        cig_reg_pred_output_table = pesudobulk_cig_reg_pred_human(cig_pred_output_table, all_db_grn_human, tf_human, training_table_cigreg_human, celltype_names)
        print(cig_reg_pred_output_table.shape)
        reg_output_file = Path(exp_filename_pred).with_suffix('.out').with_name(Path(exp_filename_pred).stem + 'cig_REG_pred_result.out')
        cig_reg_pred_output_table.to_csv(reg_output_file, index=False, sep="\t")

    elif organism_name_pred == 'hs' and assaytype_pred == 'single' and inputtype_pred == 'umicount' and exp_filename_pred is not None:
        print ("HUMAN",organism_name_pred,assaytype_pred,inputtype_pred,exp_filename_pred)
        exp_path = Path(exp_filename_pred)
        barcodes = pd.read_csv(exp_path / "barcodes.tsv", sep="\t", header=None)
        barcodes.columns = ['barcode']
        features = pd.read_csv(exp_path / "features.tsv", sep="\t", header=None)
        features.columns = ['Geneid', 'gene_symbols', 'note']
        counts_mat = io.mmread(exp_path / "matrix.mtx")
        counts_mat = counts_mat.toarray()
        counts_mat = np.matrix(counts_mat.transpose())
        # create anndata object
        adata = anndata.AnnData(counts_mat, obs=barcodes['barcode'].tolist(), var=features)
        adata.obs.index = barcodes['barcode'].tolist()
        adata.var.index = features['Geneid'].tolist()
        adata.var_names_make_unique()##REMOVE DUPLICATES
        adata_single_cell_cig = cig_pred_singlecell_human(adata, features, all_seq_features_human, training_table_human, str(exp_path))
        output_h5ad = exp_path / "_cig_matrix_out.h5ad"
        adata_single_cell_cig.T.write_h5ad(output_h5ad)

    elif organism_name_pred == 'mm' and assaytype_pred == 'bulk' and exp_filename_pred is not None:
        print ("MOUSE",organism_name_pred,assaytype_pred,inputtype_pred,exp_filename_pred)
        exp_matrix = pd.read_csv(exp_filename_pred, sep="\t")
        celltype_names = exp_matrix.columns.to_list()
        print(exp_matrix.shape)
        cig_pred_output_table = pesudobulk_cigpred_mouse(inputtype_pred, exp_matrix, all_seq_features_mouse, gene_name_length_mouse, training_table_mouse)
        first_column_symbol = cig_pred_output_table.pop('symbol')
        cig_pred_output_table.insert(0, 'symbol', first_column_symbol)
        output_file = Path(exp_filename_pred).with_suffix('.out').with_name(Path(exp_filename_pred).stem + 'cig_pred_result.out')
        cig_pred_output_table.to_csv(output_file, index=False, sep="\t")
        
        cig_reg_pred_output_table = pesudobulk_cig_reg_pred_mouse(cig_pred_output_table, all_db_grn_mouse, tf_mouse, training_table_cigreg_mouse, celltype_names)
        reg_output_file = Path(exp_filename_pred).with_suffix('.out').with_name(Path(exp_filename_pred).stem + 'cig_REG_pred_result.out')
        cig_reg_pred_output_table.to_csv(reg_output_file, index=False, sep="\t")

    elif organism_name_pred == 'mm' and assaytype_pred == 'single' and inputtype_pred == 'umicount' and exp_filename_pred is not None:
        print ("MOUSE",organism_name_pred,assaytype_pred,inputtype_pred,exp_filename_pred)
        exp_path = Path(exp_filename_pred)
        barcodes = pd.read_csv(exp_path / "barcodes.tsv", sep="\t", header=None)
        barcodes.columns = ['barcode']
        features = pd.read_csv(exp_path / "features.tsv", sep="\t", header=None)
        features.columns = ['Geneid', 'gene_symbols', 'note']
        counts_mat = io.mmread(exp_path / "matrix.mtx")
        counts_mat = counts_mat.toarray()
        counts_mat = np.matrix(counts_mat.transpose())
        # create anndata object
        adata = anndata.AnnData(counts_mat, obs=barcodes['barcode'].tolist(), var=features)
        adata.obs.index = barcodes['barcode'].tolist()
        adata.var.index = features['Geneid'].tolist()
        adata.var_names_make_unique()##REMOVE DUPLICATES
        adata_single_cell_cig = cig_pred_singlecell_mouse(adata, features, all_seq_features_mouse, training_table_mouse, str(exp_path))
        output_h5ad = exp_path / "_cig_matrix_out.h5ad"
        adata_single_cell_cig.T.write_h5ad(output_h5ad)


if __name__ == "__main__":
    main()
    