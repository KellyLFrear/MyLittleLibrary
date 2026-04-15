## Step 1: Download Wikipedia Articles, Process The Text To Clean It And Remove Unwanted Articles, And Load Vocabulary Lists For Different Proficiency Levels 
from scripts.download_wiki import download_wiki
from scripts.preprocess_wiki import preprocess_wiki
from scripts.analyze_articles import analyze_articles

# Downloads The Wikipedia Articles
download_wiki()

# Processes And Cleans The Wikipedia Articles
preprocess_wiki()

# Analyzes The Wikipedia Based On The Vocabulary Lists For Different Proficiency Levels
analyze_articles("beginner")
analyze_articles("intermediate")
analyze_articles("advanced")

