# Preprocessing Data
## Images

Processo de transformar uma imagem bruta em um array [data, m_examples] e os labels em [1, m_examples] 

Passos: 

1. Image Generator:
   1. Para o projeto text background, criei imagens de diferentes tonalidades de vermelho, azul e verde. Mas essa poderia ser a etapa de buscar um dataset com as imagens que precisamos
   2. Agora seria a hora de organizar e limpar o dataset de conteúdo com falhas
   
2. Data Manipulation:
   1. Acessar os locais de localização da data e inicilizar os parâmetros das categorias e rearranjo de data que seão usados
   2. Criar os dados de training data e test data, dado resize apropriado e anexando a um vetor
   3. Feito isso, salvar os dados obtidos, podendo usar o pickle ou numpy
   4. Dar um resize final

3. Check Data
   1. Checar os dados 
   2. Standardize dataset

