# O que eu aprendi

English: [../LEARNINGS.md](../LEARNINGS.md)

Cada lição diz de onde veio. "Medido" quer dizer um número das nossas próprias rodadas; "conselho" quer dizer algo que lemos ou nos disseram e não testamos.

## Sobre o treino

1. **Julgue pela loss de validação, nunca pela de treino.** Medido na Fase 1: a validação chegou ao mínimo de 1,469 e subiu para 1,679 enquanto o treino caía a 0,625. Guarde o checkpoint de melhor validação e pare depois de algumas avaliações sem melhora.
2. **Losses só se comparam com os mesmos dados e o mesmo tokenizer.** Os números do modelo de caracteres (1,4) e os do BPE (3,1) não são comparáveis.
3. **O otimizador pesou muito em escala pequena.** Medido: o Muon venceu o AdamW por 0,86 de loss no piloto de 52M tokens, com cerca de 4% menos velocidade. Ressalva: pilotos curtos exageram diferenças, e o AdamW só foi testado com duas learning rates.
4. **Conserte o código antes do treino longo.** A revisão independente achou problemas reais (dados vistos com repetição, uma atualização do Muon que favorecia um bloco, um conjunto de validação que dependia do tamanho do lote, falta de retomada). A retomada foi testada e dá pesos idênticos bit a bit, e o treino principal foi parado e retomado uma ou mais vezes sem problemas.
5. **Tokens por parâmetro.** 3,0 bi de tokens para 110M de parâmetros dão cerca de 27 tokens por parâmetro, acima da regra prática de 20, com uma passada pelos dados: a validação caiu até o fim (sem overfitting).
6. **Velocidade é, em grande parte, não desperdiçar a GPU.** O torch.compile deu +38% de velocidade e -38% de memória no piloto. A loss é somada na GPU com uma sincronização por passo. O treino chegou a 33,0 TFLOP/s, cerca de dois terços do pico de matmul.
7. **Contagem honesta de FLOPs.** A atenção causal faz metade do trabalho; contá-la inteira exagerava a velocidade em 8 a 10%.

## Sobre o ajuste fino (SFT)

8. **O SFT ensina formato, não conhecimento.** Depois do SFT o modelo cumprimenta, responde e para, mas os fatos continuam pouco confiáveis.
9. **Mascare o pedido, mantenha o token de fim.** Loss só nos tokens da resposta, e o token de fim de texto faz parte da resposta para o modelo aprender a parar.
10. **Remova duplicatas antes de dividir** para que cópias da validação não estejam no treino. Foram removidos 20.213 primeiros pedidos repetidos.
11. **Descarte, não corte, exemplos longos demais.** Cortar apagaria o token de fim e ensinaria que as respostas terminam em pontos aleatórios.
12. **Mais exemplos diferentes valem mais que mais passadas pelos mesmos.** O overfitting vem da repetição. A rodada final viu 446 mil chats distintos duas vezes; a segunda época acrescentou só 0,057.
13. **Teste a learning rate numa rodada mais curta.** 3e-4 ganhou com 1,630 contra 1,706 e 1,761, mas era a maior testada, então o ótimo pode ser mais alto.
14. **A loss de validação não mostra deriva.** Leia amostras. Modelos antigos do teste entravam em loop ("Read the text" x15); o final não.
15. **Etapas pequenas e específicas precisam de métrica específica.** O script guardava o checkpoint de melhor validação geral, que é 98% chats comuns, então escolheu o passo 200, antes de o nome ser aprendido. Para uma etapa pequena, use o último checkpoint ou meça o alvo.
16. **Repetir um conjunto minúsculo muitas vezes tem preço.** Chats de identidade x8 ensinaram o nome, mas também foram decorados (loss de treino ~1,0, validação subindo de 1,488 para 1,516) e fizeram o modelo recitar a identidade em mensagens curtas sem relação. A correção de fundo, não feita, é ter dados de conversa casual mais variados e menos cópias da identidade.
17. **O modelo copia a si mesmo.** O histórico do chat está no prompt, então um modelo pequeno repete as próprias respostas anteriores; temperatura baixa deixa os loops grudentos.

## Sobre a geração de texto

18. **Temperatura baixa gera loops.** Medido no modelo base: de 0,3 a 0,5 ele repetia as mesmas frases e sustentava fatos errados com confiança (a água ferve a 15 C). A faixa útil foi de cerca de 0,7 a 0,9 com top-k de 40 a 50 no modelo base.
19. **Resolva a repetição na hora do chat antes de retreinar.** Pontue as respostas pelas frases repetidas de 4 tokens, rejeite e tente de novo mais quente, corte no primeiro "###". Offline: 112 frases repetidas para 0. Só a proibição de n-gramas repetidos pode deixar as palavras estranhas quando o modelo não tem mais o que dizer, então ela é opcional.
20. **Digitar no prompt errado.** Um comando de terminal colado no chat é só texto para o modelo; ele o continua. Nada é executado.

## Sobre medir

21. **Amostras pequenas mentem.** 100 perguntas do HellaSwag deram 44%; as 10.042 completas deram 32,3%. Informe o tamanho da amostra e o erro padrão.
22. **Teste o código de avaliação** contra uma referência lenta um-a-um (diferença de 1e-5).
23. **O resultado honesto é misto.** O modelo base fica acima do acaso em HellaSwag, ARC-Easy e PIQA e no acaso em ARC-Challenge e Winogrande. O SFT baixa as notas em texto puro em alguns pontos. O modelo base fica quase igual ao GPT-2 124M no HellaSwag (31,1 informado para o GPT-2, outro código de avaliação, só aproximadamente comparável).

## Sobre dados e licenças

24. **Dados sintéticos herdam os termos de quem os gerou.** 57% das conversas de chat vieram do Llama 3.1 405B, então o modelo leva o nome Llama, o aviso "Built with Llama", a licença e um arquivo NOTICE. O Dolly é CC BY-SA 3.0 e o FineWeb-Edu é ODC-By: a atribuição é obrigatória. Não é aconselhamento jurídico.
25. **Não republique texto da web.** Aponte para a página do conjunto de dados.
26. **Dados sintéticos de personagem fazem modelos pequenos inventarem histórias.** Brincar de teatro com o modelo ("sou seu pai") o fez inventar um pai chamado David.

## Sobre o jeito de trabalhar

27. **Segredos.** Um token do Hugging Face foi colado num log do terminal; foi revogado na hora. Use variáveis de ambiente, mantenha o `.env` fora do git e nunca cole tokens em chats ou logs.
28. **Confira o que você entregou.** Arquivos enviados ao PC às vezes chegavam na versão anterior. A solução: esperar e comparar byte a byte.
29. **Rode um comando por vez.** Colar um comando com um chat ainda aberto o envia ao modelo como mensagem.
30. **Documentar no caminho é barato, reconstruir depois não é.** Cada fase tem um log datado; este repositório foi montado a partir deles.

## Ideias, não testadas

Só conselhos ou planos; nada foi construído nem medido.

- Treinar junto com outros PCs: o treino comum com paralelismo de dados exige conexões rápidas entre as máquinas, e a internet é lenta demais. Existem métodos de pouca comunicação (DiLoCo, OpenDiLoCo, INTELLECT-1) para isso; um teste com duas máquinas é uma ideia.
- Uma rodada de polimento (cerca de 150 chats casuais variados, menos cópias da identidade, escolher o checkpoint por uma métrica de identidade mais conversa casual).
- Learning rate 5e-4 no SFT; aritmética com rascunho usando dados gerados por programa; uma demo de recuperação de documentos (RAG) sobre as notas do projeto; mais tokens de pré-treino (a amostra de 10B) ou um modelo maior (cerca de 325M com contexto 2.048).
- Um modelo que "pensa" foi perguntado, mas não tentado.

Veja também: [jornada](JORNADA.md), [decisões](DECISOES.md), [glossário](GLOSSARIO.md).
