# CluHTM - Semantic Hierarchical Topic Modeling based on CluWords

This is the code for the paper:

Felipe Viegas, Washington Cunha, Christian Gomes, Antônio Pereira, Leonardo Rocha and Marcos Goncalves. CluHTM - Semantic Hierarchical Topic Modeling based on CluWords (ACL2020)

Python3 requirements:
- numpy
- scipy
- gensim
- pandas
- scikit-learn
- joblib
- matplotlib
- prettytable
- nimfa
- pillow
- seaborn

Word Embedding exploited:
- [FastText](https://fasttext.cc/docs/en/english-vectors.html) (Pre-trained Word Embedding)

Once installed, setup the paths in the startup file :

```main.py```

Build docker container:

```docker build -t cluhtm <project_path>```

Run docker container:

```docker run --rm --name cluthm -v <project_path>:/cluhtm -i -t cluhtm /bin/bash```

To run the code:

```python3 main.py -d <dataset>```

For more information about building and running a docker container, see: https://docs.docker.com/


### Cite
Conference paper:
```@inproceedings{viegas2019cluhtm,
title={CluHTM - Semantic Hierarchical Topic Modeling based on CluWords},
author={Viegas, Felipe and Cunha, Washington and Gomes, Christian and  Pereira Antonio and Rocha, Leonardo and Gonçalves, Marcos André},
booktitle={The 58th Annual Meeting of the Association for Computational Linguistics (ACL ’20)},
year={2020},
organization={ACL}
}```
