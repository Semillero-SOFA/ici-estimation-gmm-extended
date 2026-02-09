#!/usr/bin/env python
# coding: utf-8

# Libraries
import os
import argparse

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import scipy as sp
from scipy.io import loadmat

#import sofa
from utilities import sofa, gmm_utils

import json
import csv
import pandas as pd

import re

# Globals
# LOCAL_ROOT = sofa.find_root()
# GLOBAL_ROOT = LOCAL_ROOT.parent
# DATABASE_DIR = f"{GLOBAL_ROOT}/databases"
# GLOBAL_RESULTS_DIR = f"{GLOBAL_ROOT}/results"
FILENAME = os.path.basename(__file__)[:-3]

# Path to the database (with Matlab Files)
DATABASE_DIR = R"D:\Semillero SOFA\Data32GBd\Data_Mat\BetterNames"
# Output path for gmm features extracted
GLOBAL_RESULTS_DIR = "D:/Semillero SOFA/gmm_32GBd_feats"
# Create a logger for this script
logger = sofa.setup_logger(FILENAME)

# Create results directory if it doesn't exist
RESULTS_DIR = f"{GLOBAL_RESULTS_DIR}/Feats_init_centroids"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def initialize_histograms():
    """
    Initialize a nested dictionary to store histogram data.
    Returns:        dict: Nested dictionary with 6 levels deep.
    """
    # Return 6 levels deep dictionary
    return defaultdict(
        defaultdict(
            defaultdict(
                defaultdict(
                    defaultdict(
                        defaultdict(list).copy
                    ).copy
                ).copy
            ).copy
        ).copy
    )


def read_data(folder_rx: str) -> dict:
    """
    Read constellation diagram of 32 Gbaud database from .mat files in the specified folder structure.

    Args:
        folder_rx (str): Path to the root folder containing received data.
    Returns:
        dict: Nested dictionary containing the constellation data organized by
              distance, power, spacing, OSNR, song, and orthogonal polarization.
    """
    data = initialize_histograms()

    for dist_pow in os.listdir(folder_rx): #0km0dbm, 2700dbm, 2709dbm
        if not os.path.isdir(os.path.join(folder_rx, dist_pow)):
            continue
        logger.info(f"Reading {dist_pow}")
        for spac in os.listdir(os.path.join(folder_rx, dist_pow)):
            logger.info(f"Reading {dist_pow}/{spac}")
            consts = os.listdir(os.path.join(folder_rx, dist_pow, spac))
            for const in consts:
                if const.endswith("xlsx"):
                    continue
                song, orth, osnr, spacing, distance, power = const.split("_")
                power = power.split(".")[0]
                spacing = spacing.replace("p", ".")
                osnr = osnr.replace("p", ".")
                mat = sp.io.loadmat(os.path.join(folder_rx, dist_pow, spac, const))
                data[distance][power][spacing][osnr][song][orth] = mat["rconst"][0]
    return data

def load_32gbaud_db(
    path: Path, full: bool = False, subfolder: str = "0km_0dBm"
    ) -> dict:
    """
    Load the 32 Gbaud database from the specified path.
    Args:
        path (Path): Path to the root folder containing the database.
        full (bool): If True, load the entire database; otherwise, load only the specified subfolder.
        subfolder (str): Name of the subfolder to load when full is False.
    Returns:
        dict: Nested dictionary containing the constellation data organized by
              distance, power, spacing, and OSNR.
    """
    # Check subfolder parameter
    if subfolder not in ["0km_0dBm", "270km_0dBm", "270km_9dBm"]:
        raise ValueError("Invalid subfolder name.")

    METADATA_PATTERN = re.compile(
        r"Song\d+_[XY]_"  # Match the prefix (e.g., Song1_X or Song1_Y)
        # Match OSNR (integer or decimal with 'p')
        r"(?P<OSNR>[\d]+(?:p[\d]+)?)dB_"
        # Match Spacing (integer or decimal with 'p')
        r"(?P<Spacing>[\d]+(?:p[\d]+)?)GHz_"
        # Match Distance (integer or decimal with 'p')
        r"(?P<Distance>[\d]+(?:p[\d]+)?)km_"
        # Match Power (integer or decimal with 'p')
        r"(?P<Power>[\d]+(?:p[\d]+)?)dBm"
    )
    dfs = []
    data = initialize_histograms()
    for directory in path.iterdir():
        if not directory.is_dir():
            continue
        # Check directory name when not loading whole DB
        if not full and directory.name != subfolder:
            continue
        for file_path in directory.rglob("*.mat"):
            match = METADATA_PATTERN.search(file_path.name)
            if not match:
                raise ValueError(
                    f"File name {file_path.name} does not match the expected pattern."
                )

            metadata = {
                key: float(value.replace("p", "."))
                for key, value in match.groupdict().items()
            }

            mat = loadmat(file_path)
            mat = mat["rconst"][0]


            # Store in nested dictionary
            data[metadata["Distance"]]\
                [metadata["Power"]]\
                [metadata["Spacing"]]\
                [metadata["OSNR"]] = mat
    return data


def calculate_gmm_2d(input_data, n_componentes, covariance_type, init_centroids=False):
    """
    Calculate a 2D Gaussian Mixture Model (GMM) for the given input data.

    Args:
        input_data (np.ndarray): Input data for GMM fitting.
        n_componentes (int): Number of Gaussian components.
        covariance_type (str): Type of covariance to use ("full", "diag", or "spherical").
        init_centroids (bool): Whether to initialize centroids. The centroids are initialized according to the 16-QAM constellation.
        When n_componentes is different from 16, each centroid will be repeated.

    Returns:
        GaussianMixture: Fitted GMM model.
    """
    if init_centroids:
        # Define 16-QAM centroids
        qam_16_centroids = np.array([
            [-3, -3], [-3, -1], [-3, 1], [-3, 3],
            [-1, -3], [-1, -1], [-1, 1], [-1, 3],
            [1, -3], [1, -1], [1, 1], [1, 3],
            [3, -3], [3, -1], [3, 1], [3, 3]
        ])
        # Repeat centroids if n_componentes > 16
        if n_componentes > 16:
            repeats = n_componentes // 16 + (n_componentes % 16 > 0)
            centroids = np.tile(qam_16_centroids, (repeats, 1))[:n_componentes]
        else:
            centroids = qam_16_centroids[:n_componentes]
        gm_kwargs = {
            "n_components": n_componentes,
            "covariance_type": covariance_type,
            "init_params": "kmeans",
            "means_init": centroids,
            "random_state": 42
        }
    else: 
        gm_kwargs = {
            "n_components": n_componentes,
            "covariance_type": covariance_type,
            "random_state": 42
        }
    return gmm_utils.calculate_gmm(input_data, gm_kwargs)

def process_channel(x_ch, n_componentes, covariance_type, init_centroids=False):
    """
    Process a single channel scenario and calculate its GMM.
    Args:
        x_ch (np.ndarray): Channel data (it's a 2D array with real and imaginary parts).
        n_componentes (int): Number of Gaussian components.
        covariance_type (str): Type of covariance to use ("full", "diag", or "spherical").
        init_centroids (bool): Whether to initialize centroids. The centroids are initialized according to the 16-QAM constellation.
        GaussianMixture: Fitted GMM model for the channel.
    """
    input_data = np.vstack((x_ch.real, x_ch.imag)).T
    gm_2d = calculate_gmm_2d(input_data, n_componentes, covariance_type, init_centroids=init_centroids)

    return gm_2d


def process_all_channels(df, spacing, osnr,X_chs, bins, limits, n_componentes, covariance_type, init_centroids=False):
    """
    Iterate over all spacing and OSNR values from dataset, process all of them and
    extract GMM parameters (weights, means and covariances) into a dataframe.

    Args:
        df (pd.DataFrame): DataFrame to store GMM parameters.
        spacing (float): Spacing value.
        osnr (float): OSNR value.
        X_chs (list): List of channel data arrays.
        n_componentes (int): Number of Gaussian components.
        covariance_type (str): Type of covariance to use ("full", "diag", or "spherical").
    Returns:
        pd.DataFrame: Updated DataFrame with GMM parameters, spacing, and OSNR.
    """
    for x_ch in X_chs:
        gm_2d = process_channel(x_ch, n_componentes, covariance_type, init_centroids=init_centroids)
        # Extract GMM parameters
        weights = gm_2d.weights_.tolist()
        means = gm_2d.means_.flatten().tolist()
        covariances = gm_2d.covariances_.flatten().tolist()

        row = {}
        for j in range(n_componentes):
            row[f"gauss_{j+1}_w"] = weights[j]
            row[f"gauss_{j+1}_m_1"] = means[2*j]
            row[f"gauss_{j+1}_m_2"] = means[2*j + 1]
            if covariance_type == "full":
                row[f"gauss_{j+1}_cov_11"] = covariances[4*j]
                row[f"gauss_{j+1}_cov_12"] = covariances[4*j + 1]
                row[f"gauss_{j+1}_cov_21"] = covariances[4*j + 2]
                row[f"gauss_{j+1}_cov_22"] = covariances[4*j + 3]
            elif covariance_type == "diag":
                row[f"gauss_{j+1}_cov_1"] = covariances[2*j]
                row[f"gauss_{j+1}_cov_2"] = covariances[2*j + 1]
            elif covariance_type == "spherical":
                row[f"gauss_{j+1}_cov"] = covariances[j]
        row["spacing"] = spacing
        row["osnr"] = osnr


        # Add to dataframe
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def calculate_gmm_and_histograms(data, init_centroids=False) -> dict:
    """
    Calculate GMM models and histograms (not included) for the provided dataset.
    In this case, it's a 32 GBd database, and save into a csv file the GMM parameters.

    Args:
        data (dict): Nested dictionary containing the constellation data.
    Returns:
        dict: Nested dictionary containing histogram data (not included here).
    """
    #histograms_hist = initialize_histograms()
    histograms_gmm = initialize_histograms()
    bins = 128
    limits = [-5, 5]
    total_componentes = [i for i in range(16, 64+1, 8)]
    total_covs = ["diag", "spherical"]
    for distance, powers in data.items():
        for power, spacings in powers.items():
            for n_componentes in total_componentes:
                # Hacer directorio para cada n_componentes
                RESULTS_DIR_N_COMPS = f"{RESULTS_DIR}/{distance}{power}/{n_componentes}_gaussians"
                Path(RESULTS_DIR_N_COMPS).mkdir(parents=True, exist_ok=True)
                for covariance_type in total_covs:
                    file_models_gmm = f"{RESULTS_DIR_N_COMPS}/models32_gmm_{covariance_type}.json"
                    file_models_gmm_csv = f"{RESULTS_DIR_N_COMPS}/models32_gmm_{covariance_type}.csv"
                    #crear dataframe para csv
                    df = pd.DataFrame()


                    #si no existe el archivo, calcular
                    if os.path.exists(file_models_gmm):
                        logger.info(f"Skipping {n_componentes} components and {covariance_type} covariance, file exists")
                    else:
                        for spacing, osnrs in spacings.items():
                            for osnr, songs in osnrs.items():
                                for song, orths in songs.items():
                                    for orth, X_rx in orths.items():
                                        logger.info(f"Calculating GMM for: {distance}/{power}/{spacing}/{osnr}/{song}/{orth}")
                                        X_chs = gmm_utils.split(X_rx, 3)
                                        df = process_all_channels(df, spacing, osnr, X_chs, bins, limits, n_componentes, covariance_type, init_centroids=init_centroids)

                        df.to_csv(file_models_gmm_csv, index=False)

    return dict(histograms_gmm)

def main():
    folder_rx = f"{DATABASE_DIR}"

    # Read received data
    logger.info("Reading data...")
    data = read_data(folder_rx)
    print("Data read succesfully")
    gmm_utils.calc_once(
            "models_tuple", calculate_gmm_and_histograms, {"data": data})
    logger.info("Features calculated successfully")

if __name__ == "__main__":
    main()