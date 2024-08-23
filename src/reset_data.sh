#!/bin/bash
''' A exécuter dans mycar/ pour supprimer les images enregistrées par la PiCam '''
# Chemin vers le dossier data
DATA_DIR="data/"

# Sauvegarder les données existantes
BACKUP_DIR="data_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
cp -r $DATA_DIR/* $BACKUP_DIR/

# Vider le dossier images
rm -r $DATA_DIR/images/*
mkdir -p $DATA_DIR/images

# Supprimer les fichiers de catalogues et le manifeste
rm $DATA_DIR/catalog_*.catalog
rm $DATA_DIR/catalog_*.catalog_manifest
rm $DATA_DIR/manifest.json

echo "Dossier data réinitialisé. Une sauvegarde a été créée dans $BACKUP_DIR."