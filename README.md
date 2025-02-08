# 📌 Desafio cientista de dados
Desafio proposto por empresa para a criação de uma plataforma de alugueis temporários na cidade de New York. O projeto tem como objetivo  a resolução de problemas de negócios, análise de dados e aplicação de modelos preditivos.

## 🛠️ Instalação
Clone este repositório e instale as dependências:

!Importante tem o conda ativado em sua variavel de ambiente

```bash
git clone https://github.com/seuusuario/nome-do-repositorio.git
cd nome-do-repositorio
conda activate seu_ambiente
```

## 📊 Análise Exploratória
O notebook contendo as análises estátiticas e EDA podem sem encontradas no diretório com a extensão ```.ipynb```

## 🤖 Modelagem
Primeiro carregue o ```one_hot_encode.pll``` 
Segundo carregue```scaler.pkl``` para normalizar os dados 
Terceiro carregue o modelo treinado ```rf_modelo_precificacao.pkl```, assim:
```python

import pandas as pd
import pickle

with open('rf_modelo_precificacao.pkl', 'rb') as f:
    modelo_carregado = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler_carregado = pickle.load(f)

# aplique as mesmas transformações ao novo apartamento antes da previsão
novo_apartamento_encoded = one_hot_encoder.transform(novo_apartamento)
novo_apartamento_scaled = scaler_carregado.transform(novo_apartamento_encoded)

# fazer a previsão
preco_log = modelo_carregado.predict(novo_apartamento_scaled)
# reverte a transformação logarítmica
preco_real = np.expm1(preco_log)  

print(f'o preço estimado para o apartamento é: ${preco_real[0]:.2f}')
```
## 
