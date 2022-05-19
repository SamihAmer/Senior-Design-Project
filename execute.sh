#!/bin/bash
echo PLEASE ENTER PATIENT ID. THIS SHOULD MATCH THE TITLE OF THE .MOV FILE
read varname
echo THANK YOU
pip install pipreqs
pipreqs .
pip install -r requirements.txt
python3 Iris_Face_Mesh.py
mkdir $varname
mv *.png $varname 
mv *.csv $varname 
rm -rf requirements.txt
