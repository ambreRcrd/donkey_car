#!/bin/bash
''' A exécuter dans sandbox/src/ pour supprimer les logs de la simulation '''
# Chemin vers le dossier log
LOG_DIR="../dataset/log/*.jpg"

# Vider le dossier images
rm -r $LOG_DIR/log/*
mkdir -p $LOG_DIR/log/*

echo "Dossier log réinitialisé."