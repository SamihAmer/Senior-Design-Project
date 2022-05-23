# Senior Design NEquest Virtual Neurological Assistant 
- This is the github repository for the code of team CSANS for our senior design project.  
- Tweaks were made to the confidence value increased from 0.8 to 0.9, this may result in the need for ideal conditions to measure data. This can be reversed.
- PLEASE ENSURE that miniconda3 is installed to your HOME directory. If anaconda3 is desired, it will require different implementation
- cd /Users/PUT_USERNAME_HERE/miniconda3/lib/python3.9/site-packages/mediapipe/python/solutions
- Current algorithm runs in O(2n)
# Data Collection Protocol
1) First, download the most recent code repository from github. This can be done either via "git clone https://github.com/SamihAmer/project" or downloading it as a zip file. You should now have a folder either named "project"  or "project-main" which houses our most recent code.
2) Drag/drop the patient .mov file into the "project" folder. Please make sure that the patient .mov file is labeled based on a unique Patient ID. 
3) Open the terminal/command prompt and navigate into the "project" directory 
4) Run "sh execute.sh" when in the "project" directory. This script will install all the needed dependencies and run the main python code. 
5) Enter the Patient ID into the terminal, this should match the Patient ID of the .mov file
6) After this command is run, you will see a playback of the .mov file. Once the code is finished running a graph will pop up which you can exit. 
7) Navigate back to the "project" folder and you will now see a newly created ZIP file which you can send to us. 
8) There is also a folder that is labeled with the Patient ID and it houses the unzipped contents of the ZIP file for your inspection if necessary.  
9) Remove the old .mov file from the directory and you can now replace it with the .mov file of a new patient.

