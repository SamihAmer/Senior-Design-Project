#!/bin/bash
count=`ls -1 *.mov 2>/dev/null | wc -l`
if [ $count == 0 ]
then 
echo PLEASE PUT .mov FILE INTO project DIRECTORY AND RE-RUN execute.sh
exit 
fi 
if [ $count != 1 ]
then 
echo PLEASE ENSURE THAT ONLY ONE .mov FILE IS IN THE project DIRECTORY AND RE-RUN execute.sh
exit 
fi 
echo PLEASE ENTER PATIENT ID. THIS SHOULD MATCH THE TITLE OF THE .mov FILE
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
