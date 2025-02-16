# InterSystems IRIS Vector Search

This year, we're adding a powerful [Vector Search capability to the InterSystems IRIS Data Platform](https://www.intersystems.com/news/iris-vector-search-support-ai-applications/), to help you innovate faster and build intelligent applications powered by Generative AI. At the center of the new capability is a new [`VECTOR` native datatype](https://docs.intersystems.com/iris20241/csp/docbook/DocBook.UI.Page.cls?KEY=RSQL_datatype#RSQL_datatype_vector) for IRIS SQL, along with [similarity functions](https://docs.intersystems.com/iris20241/csp/docbook/Doc.View.cls?KEY=GSQL_vecsearch) that leverage optimized chipset instructions (SIMD). Basically, IRIS is a SQL database that's really fast, and now has vector search built in!

_Prerequisite_ - [Docker](https://www.docker.com) must be installed and running for the commands below to work!

## Quickstart

1. Clone the repo
    ```Shell
    git clone https://github.com/intersystems-community/hackathon-2024.git
    cd hackathon-2024
    ```


2. Install IRIS Community Edtion in a container. This will be your SQL database server.
    ```Shell
    docker run -d --name iris-comm -p 1972:1972 -p 52773:52773 -e IRIS_PASSWORD=demo -e IRIS_USERNAME=demo intersystemsdc/iris-community:latest
    ```
   After running the above command, you can access the System Management Portal via http://localhost:52773/csp/sys/UtilHome.csp.

3. Create a Python environment and activate it (conda, venv or however you wish) For example:

   NOTE: The DB-API driver .whl files in step 5  might only work with python 3.8 to 3.12. If you get an error while installing those files, you will have to create a virtual environment with a specific python version like "python3.12 -m venv myenv"
   
    conda:
    ```Shell
    conda create --name iris-env python=3.10
    conda activate
    ```
    venv(Mac):
    ``` Shell
    python3 -m venv iris-env
    source iris-env/bin/activate
    ```
    or

    venv (Windows):
    ```Shell
    python3 -m venv iris-env
    .\iris-env\Scripts\Activate
    ```
    or

    venv (Unix):
    ```Shell
    python -m venv iris-env
    source ./iris-env/bin/activate
    ```

4. Install packages for all demos -- *Note*: This command might take a while to run (as it freezes for some time which looks like its stuck):
    ```Shell
    pip install -r requirements.txt
    ```

5. Install Intersystem's DB API driver . Choose one option, based on your Operating System. Usage of the driver is subject to [`Terms and Conditions`](https://www.intersystems.com/IERTU)

    Mac OS:

    ```Shell
    pip install ./install/intersystems_irispython-5.0.1-8026-cp38.cp39.cp310.cp311.cp312-cp38.cp39.cp310.cp311.cp312-macosx_10_9_universal2.whl
    ```

    Windows AMD64:

    ```Shell
    pip install ./install/intersystems_irispython-5.0.1-8026-cp38.cp39.cp310.cp311.cp312-cp38.cp39.cp310.cp311.cp312-win_amd64.whl
    ```

    Windows 32:
    ```Shell
    pip install ./install/intersystems_irispython-5.0.1-8026-cp38.cp39.cp310.cp311.cp312-cp38.cp39.cp310.cp311.cp312-win32.whl
    ```

    Linux aarch64:
    ```Shell
    pip install ./install/intersystems_irispython-5.0.1-8026-cp38.cp39.cp310.cp311.cp312-cp38.cp39.cp310.cp311.cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    ```

    Linux x86_64:
    ```Shell
    pip install ./install/intersystems_irispython-5.0.1-8026-cp38.cp39.cp310.cp311.cp312-cp38.cp39.cp310.cp311.cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    ```

6. For [`langchain_demo.ipynb`](demo/langchain_demo.ipynb) and [`llama_demo.ipynb`](demo/llama_demo.ipynb), you need an [OpenAI API Key](https://platform.openai.com/api-keys). Create a `.env` file in this repo to store the key:
    ```
    OPENAI_API_KEY=xxxxxxxxx
    ```

7. You can run the demo notebooks using jupyter notebooks or VSCode. To run using jupyter notebooks:
    ```Shell
    pip install jupyter
    jupyter notebook
    ``` 

8. To run the demo using VSCode:
   - Open VSCode and navigate to **File -> Open Folder...**, then select the **hackathon** folder.
   - Open the notebook file you want to run from the **Demo** folder.
   - In the **top right** of the notebook, click **Select Kernel** and select **iris-env** to activate the environment that was created.



## Using the Management Portal

1. Navigate to http://localhost:52773/csp/sys/UtilHome.csp, login with username: demo, password: demo (or whatever you configured)
2. Change the namespace (on the top left) from %SYS to USER
3. On the left navigation pane, click 'System Explorer'
4. Click 'SQL' -> 'Go'
5. Here, you can execute SQL queries. You can also view the tables by clicking the relevant table on the left, under 'Tables', and then clicking 'Open Table' (above the SQL query box)

## Basic Demos

### [IRISDatabaseOperationsUsingSQL.ipynb](demo/IRISDatabaseOperationsUsingSQL.ipynb) - Recommended!

This demo uses our latest db api driver, which is more efficient.

Here, we first demonstrate how to connect to an IRIS db and carry out basic CRUD operations.

We then use IRIS Vector seach in a whishkey dataset to find whiskeys that are priced < $100 and have a taste description _similar_ to "earthy and creamy taste". This demo uses SQL for vector search.

### [langchain_demo.ipynb](demo/langchain_demo.ipynb)

IRIS now has a langchain integration as a VectorDB! In this demo, we use the langchain framework with IRIS to ingest and search through a document.

### [llama_demo.ipynb](demo/llama_demo.ipynb)

IRIS now has a llama_index integration as a VectorDB! In this demo, we use the llama_index framework with IRIS to ingest and search through a document.

## SQL Vector SearchSyntax

Here's some [`documentation`](demo/SQLSyntax.md) on of our vector search syntax. Let us know if you need any assistance with setting up SQL queries.

## Which to use?

If you need to use search with filters, use IRIS SQL. This is the most flexible way to build RAG.

If you're building a genAI app that uses a variety of langchain tools (agents, chained reasoning, api calls), go for langchain.

If you're building a simple RAG app, go for llama_index.

The fastest and easiest way to contact any InterSystems Mentor is via Slack or Discord - feel free to ask any questions about our technology, or about your project in general!


## More Demos / References:

### [NLP Queries on Youtube Audio Transcription](https://github.com/jrpereirajr/intersystems-iris-notebooks/blob/main/vector/langchain-iris/nlp_queries_on_youtube_audio_transcription_dataset.ipynb)
Uses langchain-iris to search Youtube Audio transcriptions

### [langchain-iris demo](https://github.com/caretdev/langchain-iris/blob/main/demo.ipynb)
Original IRIS langhain demo, that runs the containerized IRIS in the notebook

### [llama-iris demo](https://github.com/caretdev/llama-iris/blob/main/demo.ipynb)
Original IRIS llama_index demo, that runs the containerized IRIS in the notebook

### [InterSystems Documentation](https://docs.intersystems.com/)
Official page for InterSystems Documentation
