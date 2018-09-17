#!/bin/bash

mkdir -p bioasq6_data

cd bioasq6_data

wget -c https://archive.org/download/aueb_bioasq6_data/bioasq6_data.tar.gz
tar -zxvf bioasq6_data.tar.gz

wget -c https://archive.org/download/biomedical_idf.tar/biomedical_idf.tar.gz
tar -zxvf biomedical_idf.tar.gz

wget -c https://archive.org/download/pubmed2018_w2v_30D.tar/pubmed2018_w2v_30D.tar.gz
tar -zxvf pubmed2018_w2v_30D.tar.gz

wget -c https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz
tar -zxvf pubmed2018_w2v_200D.tar.gz
