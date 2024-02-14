# Steam-Search-Engine
Search engine based on Steam collection's games.

## Installation
Download the repository:  
`git clone https://github.com/RaffaeleTranfaglia/Steam-Search-Engine.git`  
In the main directory "Steam-Serach-Engine".  
Uncompress the dataset:  
`unzip Dataset.zip`  
Install dependecies:  
`python3 -m pip install -r requirements.txt`  
Run the setup script to create all the indexes, download nltk corpora and the AI model for sentiment analysis:  
`pyhton3 -m setup [-n <number of threads>]`  

Observations: 
- AI model adopted (`j-hartmann/emotion-english-distilroberta-base`) may take several minutes to be installed
- Indexes creation time depends on the number of threads allocated when running `setup` script, with the default value (4) it takes around 3:20 hours
- Running the main program without having built the indexes 


## Usage
Run the application:
`python3 -m main`
There
