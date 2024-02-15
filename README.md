# Steam-Search-Engine
Search engine based on Steam's collection of games.

## Installation
It is advised to download the project in a virtual environment.
On MacOs systems:
```
virtualenv venv
source venv/bin/activate
```

Download the repository:  
```
git clone https://github.com/RaffaeleTranfaglia/Steam-Search-Engine.git
```  
In the main directory "Steam-Serach-Engine".  
Uncompress the dataset:  
```
unzip Dataset.zip
```  
Install dependecies:  
```
pip install -r requirements.txt
```  
Run the setup script to create all the indexes, download nltk corpora and the AI model for sentiment analysis:  
```
pyhton3 -m setup [-t <number of threads>]
```  

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
- **BenchmarkUtilities Package**: package that contains benchmarks evaluation utils and the list of benchmark queries
- **GUI Package**: package that contains the GUI code
- **indexdir directory**: directory that contains both versions of the index
- **MainImplementation Package**: package that contains the main implementation of the searcher
- **TextUtils Package**: package that contains text preprocessing code and the documents' indexer
- **footage Directory**: directory that contains assets
- **Benchmarks.ipynb file**: notebook to run benchmarks
- **Dataset.zip file**: zip of the whole dataset
- **main.py file**: main file of the search engine
- **requirements.txt**: list of packages required to run the program
- **setup.py**: script which downloads the necessary nltk corpora, downloads the RoBERTa model and creates both version of the index

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
The GUI (Graphical User Interface) is the front-end of the search engine.  
It will appear as soon as the main module is executed: `python3 -m main`.
![gui1](/footage/gui1.png)
![gui2](/footage/gui2.png)
Widgets usage:
![gui3](/footage/gui3.png)

## Benchmarks
The `Benchmarks.ipynb` notebook shows many performance measures of each version of the project. Every benchmark is the result of testing a version on a benchmark queries set, defined as a json file in `BenchmarkUtilities` package.  
Below are two measures extracted from the notebook.  
### Average Precision
Average precision at each standard recall level across all queries of the benchmark queries set.  
Evaluates overall system performance on a query corpus.  
![average_precision](/footage/average_precision.png)
### Mean Average Precision (MAP)
The average of the average precisions across multiple queries.  
![map](/footage/map.png)

## Dataset
The dataset adopted is contained in `Dataset.zip`. It is a directory of JSON files where every file is a game.  
Every game file is a dictionary, every field of the dictionary is defined below:  
- **app_id**: identifier of steam games
- **name**: game's title
- **release_date**: release date of the game
- **developer**: list of the game's developers
- **publisher**: list of the game's publishers
- **platforms**: list of platforms where the game is available
- **required_age**: not used
- **categories**: list of game's categories
- **genres**: list of game's genres
- **tags**: list of game's tags
- **achievements**: not used
- **positive_ratings**: number of positive ratings (_"Recommended"_)
- **negative_ratings**: number of negative ratings (_"Not Recommended"_)
- **price**: game's price
- **description**: game's description
- **header_img**: link to the game's header image
- **minimum requirements**: minimum requirements to run the game
- **recommended requirements**: recommended requirements to run the game, not always present
- **reviews**: list of the game's reviews, each composed of:
    - **review_text**: text of the review
    - **review_score**: "1" if positive, else "-1"

The data used to create the Dataset folder are based on Steam games and reviews, and provided from the union of the following datasets after a data cleaning process: [source1](https://www.kaggle.com/datasets/nikdavis/steam-store-games), [source2](https://www.kaggle.com/datasets/andrewmvd/steam-reviews).  
It contains most of the Steam games released within May 2019, most of the games released within 2017 contain reviews. Each game has at most 150 reviews due to indexing times' reasons.

## Possible Future Improvements
- Fine tuning of the AI model used for sentiment analysis
- Query expansion

## Authors
- Raffaele Tranfaglia
- Samuele Tondelli
