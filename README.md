# Steam-Search-Engine
Search engine based on Steam's collection of games.

## Installation
It is advised to download the project in a virtual environment.
On MacOs systems:
```
python3 -m virtualenv venv
source venv/bin/activate
```

Download the repository:  
`git clone https://github.com/RaffaeleTranfaglia/Steam-Search-Engine.git`  
In the main directory "Steam-Serach-Engine".  
Uncompress the dataset:  
`unzip Dataset.zip`  
Install dependecies:  
`python3 -m pip install -r requirements.txt`  
Run the setup script to create all the indexes, download nltk corpora and the AI model for sentiment analysis:  
`pyhton3 -m setup [-t <number of threads>]`  

Observations: 
- AI model adopted (`j-hartmann/emotion-english-distilroberta-base`) may take several minutes to be installed;
- Indexes creation time depends on the number of threads allocated when running `setup` script, with the default value (4) it takes around 3 and a half hours;
- If the main program is executed without having previously create indexes, the index corresponding to the launched version is built before execution;
- Requirements are related to a MacOs system, on other devices they may not work properly, in those cases it is advised to remove the requirements which raise errors and/or download the ones related to the current system.

> Due to the time took for creating indexes, is highly recommended to download the pre-created indexes provided by the release of the project

## Usage
Run the search engine:  
`python3 -m main [-s <sentiment version> -t <number of threads>]`  
Options:  
- `-s | --sentiment` takes as argument the chosen version of sentiment analysis:
  - `false` → Base version
  - `av` → Sentiment analysis version, each game sentiment values is the average of its reviews sentiment values
  - `inav` → Sentiment analysis version, each game sentiment values is the inverted neutral weighted average of its reviews sentiment values
- `-t | --threads` takes as argument the number of threads used to build indexes (default value = 4)

### Project Structure

### Query Languages
Query languages supported by all the search engine versions:
- **Natural language query**: simple enumeration of words and context queries
  - e.g. *dark souls*
- **Phrase query**: retrieve documents with a specific sentence (ordered list of contiguos words)
  - e.g. *"dark souls"*
- **Boolean query**: single word queries or natural language queries connected by boolean operators (OR, AND)
  - e.g. *dark OR souls*
  - e.g. *Valve OR (Id Software)*
- **Pattern matching query**: query that match text rather than word tokens
  - e.g. *dark\**

Query language for sentiment queries:  
A normal query followed by *\\sentiment[]*, the square brackets contain the sentiment query
  - e.g. *doom \\sentiment\[scary\]*  
Every non sentiment version will ignore the sentiment segment of the query.

### GUI

## Benchmarks

## Dataset

## Possible Future Improvements
- Fine tuning of the AI model used for sentiment analysis
- Query expansion

## Authors
- Raffaele Tranfaglia
- Samuele Tondelli
