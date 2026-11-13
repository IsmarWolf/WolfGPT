# Glossário do WolfGPT

English: [../GLOSSARY.md](../GLOSSARY.md)

Definições em linguagem simples dos termos usados enquanto construí o WolfGPT, escritas do jeito que fui aprendendo.

## Hardware e software

**GPU** - A placa de vídeo. Faz um monte de cálculos simples ao mesmo tempo, que é exatamente o que redes neurais precisam. A nossa: NVIDIA RTX 5060 Ti.

**VRAM** - A memória da própria GPU (16 GB aqui). Os pesos do modelo, os dados que estão sendo processados e o estado do treino precisam caber nela. É por isso que um modelo de 1B de parâmetros não cabe nesta placa.

**Tensor cores** - Circuitos especiais da GPU que multiplicam pequenos blocos de números de uma só vez. As altas velocidades em 16 bits vêm deles.

**CUDA** - A camada de software da NVIDIA que deixa os programas rodarem na GPU.

**Compute capability / sm_120** - A versão da arquitetura da GPU. A nossa é 12.0 (Blackwell). O PyTorch precisa trazer código compilado para ela, por isso este projeto exige uma versão com CUDA 12.8 ou mais nova.

**WSL2** - Windows Subsystem for Linux 2: um Linux de verdade rodando dentro do Windows. As ferramentas de IA rodam no Linux, usando o driver de GPU do Windows.

**venv (ambiente virtual)** - Uma pasta isolada de pacotes Python para um projeto, para que as instalações não afetem mais nada. Precisa ser ativada em cada terminal novo (`source ~/wolfgpt-env/bin/activate`); o prompt passa a começar com (wolfgpt-env).

**PyTorch** - A biblioteca Python usada para construir e treinar redes neurais na GPU.

**torch.compile** - O compilador do PyTorch para o modelo. No piloto deu +38% de velocidade e -38% de memória.

## Matemática e velocidade

**Matriz / matmul (multiplicação de matrizes)** - Uma matriz é uma grade de números. Multiplicar duas grades quer dizer: para cada célula do resultado, multiplicar uma linha de uma por uma coluna da outra e somar. Um GPT é quase só matmul repetida bilhões de vezes, por isso a medimos.

**FLOP** - Uma operação aritmética (uma multiplicação ou uma soma). Multiplicar duas matrizes N x N custa cerca de 2 x N^3 FLOPs.

**TFLOP/s** - Trilhões de FLOPs por segundo. A nossa velocidade medida em bf16 é de cerca de 50 TFLOP/s.

**Formatos de número**
- **fp32** - 32 bits por número. Preciso, mas lento.
- **fp16** - 16 bits. Mais rápido e metade da memória, mas com um alcance estreito de tamanhos, então pode estourar.
- **bf16** - 16 bits com o mesmo alcance do fp32 mas menos detalhe. Quase nunca estoura, por isso é o nosso formato de treino.
- **TF32** - entradas fp32 arredondadas durante a multiplicação. Cerca de metade da velocidade do bf16 aqui.

**MFU (aproveitamento de FLOPs do modelo)** - A fração do pico medido que o treino real usa. O treino também move dados e faz outras contas, então 30-50% é comum. O treino principal do WolfGPT usou cerca de dois terços (33,0 de cerca de 50 TFLOP/s).

## O modelo

**Parâmetros (os "110M")** - Os números aprendidos dentro do modelo. 110M quer dizer 110 milhões deles (110.119.680 exatamente). Mais dados nunca acrescentam parâmetros; só um formato diferente de modelo acrescenta.

**Pesos** - Outra palavra para os parâmetros. O treino os ajusta.

**Token** - Um pedaço de texto, mais ou menos três quartos de uma palavra. "Tokens por segundo" é a velocidade com que o treino lê texto.

**Tokenizer / BPE** - O programa que corta o texto em tokens. O BPE de bytes (byte-pair encoding) monta o vocabulário (32.768 tokens aqui) juntando repetidamente os pares de bytes mais frequentes.

**Janela de contexto (tamanho do contexto)** - Quantos tokens o modelo consegue olhar de uma vez. WolfGPT: 1.024. Qualquer coisa maior não cabe, e o modelo só foi treinado em posições até 1.024.

**GPT** - Generative Pre-trained Transformer: um modelo treinado para prever o próximo token.

**Atenção, RoPE, RMSNorm, SwiGLU, embeddings compartilhados** - A atenção deixa cada token olhar para os anteriores. O RoPE diz à atenção onde cada token está. O RMSNorm mantém os números numa faixa estável entre as camadas. O SwiGLU é uma camada feed-forward com portão (metade dos parâmetros do WolfGPT está nela). Embeddings compartilhados quer dizer que a tabela de palavras da entrada e a da saída são os mesmos números.

**Perplexidade** - e elevado à loss: mais ou menos, entre quantas palavras o modelo está dividido a cada passo. Modelo base do WolfGPT: cerca de 22,8.

**Treino** - Mostrar texto ao modelo repetidamente, medir o quanto os palpites do próximo token estão errados (a loss) e empurrar os pesos para errarem menos.

## Como o treino é organizado

**Época** - Uma passada completa por todo o conjunto de treino. O pré-treino teve menos de uma época (cada um dos 3 bi de tokens visto cerca de uma vez). O SFT usa 2 épocas. Mais passadas ajudam o aprendizado no começo, mas passadas demais fazem o modelo decorar os exemplos (overfitting). Muitos exemplos diferentes vistos uma ou duas vezes são mais seguros que poucos exemplos vistos muitas vezes.

**Passo, lote, micro-lote** - Um passo é uma atualização dos pesos. Um lote é o grupo de exemplos usado nessa atualização (32 conversas no SFT). Um micro-lote é a fatia que cabe na GPU de uma vez; os resultados de várias fatias são somados antes da atualização (acúmulo de gradiente). No SFT as fatias são dimensionadas por tokens (4.096), não por número de conversas.

**Padding** - Tokens de enchimento acrescentados aos exemplos curtos para que todos os exemplos de um lote tenham o mesmo tamanho. Eles são ignorados na loss.

**Otimizador (AdamW, Muon)** - A regra de como atualizar os pesos a partir do gradiente. O AdamW é a escolha comum. O Muon é mais novo e é usado nas matrizes internas; no piloto chegou a uma loss de 4,418 contra 5,280 do AdamW.

**Learning rate (taxa de aprendizado)** - O tamanho de cada atualização dos pesos. Alta demais: treino instável ou esquecer o que o modelo sabia. Baixa demais: lento. **Aquecimento (warm-up)**: começar pequena e subir nos primeiros passos. **Decaimento cosseno**: baixar suavemente, aqui até 10% do pico no fim. Um **teste de learning rate (sweep)** experimenta vários valores numa rodada mais curta e fica com o melhor.

**Gradient clipping / norma do gradiente** - Limitar o tamanho de uma única atualização. Um pico na norma do gradiente avisa que o treino está ficando instável (a maior vista no treino principal foi 2,28: sem picos).

**Loss de treino vs loss de validação** - A loss de treino é medida nos dados com que o modelo aprende; a de validação em dados reservados com que ele nunca treina. A validação é o número honesto.

**Overfitting** - A loss de treino continua caindo enquanto a de validação para de melhorar ou sobe: o modelo está decorando em vez de aprender (visto na Fase 1, em minutos). Guardar o checkpoint de melhor validação protege contra isso.

**Checkpoint** - Uma cópia salva do modelo (e, se quiser, do otimizador e da posição nos dados) para o treino poder continuar depois de uma queda.

**Snapshot de retomada** - Um checkpoint que guarda também o estado do otimizador, o estado dos números aleatórios e a posição nos dados, para que uma rodada interrompida continue exatamente de onde parou e termine com os mesmos pesos. Rodar o run_overnight.sh de novo usa isso automaticamente.

**Smoke test** - Uma rodada minúscula (cerca de 20 passos) que confere se tudo funciona antes de começar a rodada longa.

## Dados

**Deduplicação** - Remover exemplos repetidos. Necessária para que uma cópia de um exemplo de validação não esteja também nos dados de treino.

**Vazamento de dados** - Quando exemplos de validação (ou cópias deles) são vistos durante o treino, então a loss de validação parece melhor do que o modelo realmente é.

**Conversas descartadas** - No preparo dos dados de SFT, conversas cuja primeira pergunta e resposta juntas passam de 1.024 tokens são removidas (cerca de 2%). Cortá-las apagaria o fim da resposta, incluindo o token de fim de texto, e ensinaria ao modelo que as respostas terminam em pontos aleatórios. Turnos posteriores de um chat com várias trocas que não cabem são cortados inteiros.

**Mensagem de sistema** - Uma instrução dada ao modelo antes da mensagem do usuário (por exemplo "resuma em uma frase"). Só a mantemos nos subconjuntos de reescrita e resumo, onde a tarefa está nela.

**Dados sintéticos** - Texto de treino escrito por outro modelo de IA em vez de por humanos. Barato e abundante, mas herda os erros e o estilo desse modelo, e seus termos de uso. O smol-smoltalk é sintético; o Dolly foi escrito por humanos.

**Magpie** - Um método para gerar dados sintéticos de instrução. Um modelo de chat recebe só o começo de um turno do usuário (o modelo de texto que normalmente vem antes de uma pergunta), então ele mesmo inventa a pergunta; depois um modelo a responde. Isso gera conjuntos enormes e variados de perguntas sem escrever prompts à mão. O magpie-ultra foi feito assim com o Llama 3.1 405B-Instruct; o **smol-magpie-ultra** é a versão de conversas mais curtas preparada para modelos abaixo de 1B de parâmetros, e é a maior parte do smol-smoltalk (57% da mistura). Como o texto vem do Llama, os termos da licença dele valem para o WolfGPT.

**Upsampling / repetição de uma fonte pequena** - Mostrar uma fonte pequena mas importante várias vezes (aqui as saudações do everyday-conversations, x3, só no treino) para ela não ser afogada por outros 450 mil chats. A validação nunca é repetida.

## Tornando o modelo útil

**Modelo base** - Um modelo que só continua texto (o que o WolfGPT é depois da Fase 4). Não responde perguntas nem conversa.

**Fine-tuning (ajuste fino)** - Continuar treinando um modelo já treinado num conjunto menor e específico, então os pesos dele mudam. Serve para mudar comportamento e formato, não para acrescentar muitos fatos confiáveis. Como ensinar uma habilidade a um aluno.

**SFT (ajuste fino supervisionado)** - Ajuste fino com conversas de exemplo (uma pergunta e a resposta que queremos). É a Fase 5: transforma o modelo base num modelo de chat.

**Mascaramento da loss** - No SFT, só os tokens da resposta contam para a loss; a pergunta do usuário é dada ao modelo mas não é pontuada, então ele aprende a responder em vez de imitar perguntas.

**Template de chat** - O formato fixo de texto de uma conversa ("### User: ... ### Assistant: ..."), mais um token de fim de texto que diz ao modelo quando a resposta terminou.

**Fase de identidade** - Um ajuste fino extra e curto em conversas escritas à mão que ensinam ao modelo o nome dele e quem o criou. Repetir um conjunto minúsculo muitas vezes ensinou o nome, mas também fez o modelo decorá-lo e recitá-lo em mensagens curtas sem relação.

**LoRA** - Um ajuste fino mais barato que treina só pequenas camadas extras em vez de todos os pesos. Economiza memória; não é necessário aqui porque o modelo inteiro de 110M cabe na GPU.

**RAG (geração com recuperação de documentos)** - Na hora da pergunta, buscar numa coleção de documentos, colar os melhores trechos no prompt e deixar o modelo responder a partir deles. O modelo não é retreinado, então os fatos mudam editando os documentos. Como uma prova com consulta. Limitado aqui pelo contexto de 1.024 tokens e pela leitura fraca do modelo pequeno. Uma ideia, não construída.

**Alucinação** - Texto fluente que afirma fatos inventados. O WolfGPT faz isso muito com 110M de parâmetros (as 3.800 células do coração, o ponto de ebulição de 15 graus).

## Amostragem (como o texto é gerado)

**Temperatura** - Controla a aleatoriedade ao escolher a próxima palavra. Baixa (0,3): sempre a palavra mais segura, o que causa loops. Alta (1,0+): mais variado, mas com mais besteira. De 0,7 a 0,9 funciona melhor no modelo base.

**Top-k** - Escolher só entre as k próximas palavras mais prováveis (usamos 40-50), o que corta as palavras muito improváveis.

**Penalidade de repetição** - Uma configuração de amostragem que baixa a nota das palavras já usadas na resposta, o que reduz os loops. A nossa é uma subtração fixa, desligada por padrão.

**Proibição de n-gramas repetidos** - Proibir gerar uma frase de N tokens que já apareceu. Eliminou todas as frases repetidas de 4 tokens num teste offline, mas pode gerar palavras estranhas quando o modelo não tem mais o que dizer.

**Rejeitar e tentar de novo** - Depois de cada resposta, medir o quanto ela repete respostas anteriores; se passar de um limite (0,3), jogar fora e gerar de novo, um pouco mais quente e com menos memória. Não precisa retreinar.

## Avaliação

**Zero-shot** - Testado sem nenhum exemplo no prompt. **Benchmark** - Um conjunto de teste público com respostas conhecidas.

**HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande** - Testes de múltipla escolha de bom senso e ciências. O modelo pontua o texto de cada resposta pela log-verossimilhança e escolhe a melhor. O chute aleatório dá 25% nos três primeiros e 50% nos dois últimos.

**acc / acc_norm** - Acurácia, e acurácia com a nota de cada resposta dividida pelo tamanho dela (para respostas longas não serem punidas).

**Erro padrão** - O quanto uma nota oscilaria com outra amostra de perguntas. Cerca de 0,5 ponto no HellaSwag e 1,4 no Winogrande.

## Licenças

**Llama 3.1 Community License** - Se saídas do Llama são usadas para treinar um modelo distribuído, o nome do modelo deve começar com "Llama", mostrar "Built with Llama" e incluir a licença e um arquivo NOTICE. Não é aconselhamento jurídico.

**ODC-By, CC BY-SA** - Licenças que exigem atribuição (e, no CC BY-SA, compartilhar sob os mesmos termos). O FineWeb-Edu é ODC-By; o Dolly é CC BY-SA 3.0.

**safetensors** - Um formato de arquivo seguro para pesos de modelos (diferente dos arquivos pickle, não consegue executar código ao ser carregado).
