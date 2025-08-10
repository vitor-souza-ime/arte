# Classificação de Obras de Arte com CLIP

Este projeto utiliza o modelo CLIP da OpenAI para realizar classificação zero-shot de obras de arte, identificando o artista provável de uma pintura a partir da imagem.

---

## Descrição

O script `arte.py` baixa imagens de obras de arte famosas em domínio público e usa o modelo CLIP para comparar a similaridade da imagem com uma lista de possíveis artistas. Em seguida, retorna o artista com a maior probabilidade segundo o modelo.

### Artistas contemplados

- Leonardo da Vinci  
- Victor Meirelles  
- Pedro Américo  

---

## Como usar

### Pré-requisitos

- Python 3.7+  
- Instalar as bibliotecas necessárias:

```bash
pip install transformers torch pillow requests
````

### Executando o script

No terminal, execute:

```bash
python arte.py
```

O script irá baixar as imagens das obras definidas e mostrar para cada uma o artista mais provável e a confiança associada.

---

## Estrutura do código

* **baixar\_imagem(url)**: Baixa uma imagem a partir da URL fornecida.
* **classificar\_clip(url, candidatos)**: Classifica a imagem entre os artistas da lista `candidatos` usando o modelo CLIP.
* Dicionário `obras` contém as obras testadas com URLs diretas para imagens em domínio público.

---

## Resultados esperados

O script imprime para cada obra:

```
Obra: <Nome da obra>
Artista previsto: <Nome do artista> (<confiança em %>)
------------------------------------------------------------
```

---

## Referências

* [CLIP - OpenAI](https://github.com/openai/CLIP)
* [Transformers - Hugging Face](https://huggingface.co/docs/transformers/index)
* Imagens de obras de arte provenientes do [Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page) em domínio público.

---

## Licença

Este projeto está licenciado sob a licença MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.

---

Se quiser, posso ajudar a gerar o arquivo LICENSE padrão MIT também. Quer?
```
