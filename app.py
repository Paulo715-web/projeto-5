import json
import math
import re
import textwrap
import time

import pandas as pd
import streamlit as st
import google.generativeai as genai
from PIL import Image

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Central Inteligente para Estudantes",
    page_icon="⚡",
    layout="wide",
)

# Paleta: grafite azulado + âmbar (a cor do raio do título)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=Figtree:wght@400;500;600&display=swap');

/* A fonte vale só para textos. Não usar seletores genéricos como [class*="st-"],
   porque eles pegam os ícones do Streamlit e mostram o nome ("light_mode") no lugar do desenho. */
html, body, .stApp,
.stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp [data-baseweb="tab"], .stApp [data-testid="stMarkdownContainer"] { font-family: 'Figtree', sans-serif; }
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
h1, h2, h3 { font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }

/* Limpa elementos do Streamlit que poluem a tela */
[data-testid="stAppDeployButton"], .stAppDeployButton, footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stMain"] .block-container { max-width: 920px; padding-top: 2rem; padding-bottom: 4rem; }

/* Cabeçalho */
.hero { margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.3rem; line-height: 1.15; margin: 0 0 .4rem 0; padding: 0; }
.hero .bolt { color: #FFB020; }
.hero p { color: #9AA7B8; font-size: 1.05rem; margin: 0; }

/* Abas */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #232C3A; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: .6rem 0; }
.stTabs [aria-selected="true"] { color: #FFB020; }

/* Mensagens do chat */
.stChatMessage {
    background: #161C26;
    border: 1px solid #232C3A;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: .6rem;
}

/* Campo de digitação */
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInput"]:focus-within { border-color: #FFB020; }

/* Botões */
[data-testid="stMain"] .stButton > button {
    border-radius: 12px;
    border: 1px solid #2A3547;
    background: #161C26;
    min-height: 4.2rem;
    height: auto;
    white-space: normal;
    text-align: left;
    padding: 0.8rem 1rem;
    transition: border-color .15s, color .15s;
}
[data-testid="stMain"] .stButton > button p { white-space: normal; overflow: visible; text-overflow: clip; }
[data-testid="stMain"] .stButton > button:hover { border-color: #FFB020; color: #FFB020; }
[data-testid="stMain"] .stButton > button[kind="primary"] {
    min-height: 2.6rem; background: #FFB020; color: #14110A; border: none; font-weight: 600;
}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover { background: #FFC24D; color: #14110A; }

/* Métricas */
[data-testid="stMetric"] { background: #161C26; border: 1px solid #232C3A; border-radius: 14px; padding: 1rem 1.2rem; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# API DO GEMINI
# =====================================================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    api_conectada = True
except Exception:
    api_conectada = False

# Confira os nomes atuais dos modelos em:
# https://ai.google.dev/gemini-api/docs/models
MODELOS = {
    "Gemini 3.5 Flash-Lite (mais usos grátis)": "gemini-3.5-flash-lite",
    "Gemini 3.8 Flash (mais capaz, poucos usos grátis)": "gemini-3.8-flash",
    "Gemini 3.1 Pro (pode exigir plano pago)": "gemini-3.1-pro-preview",
}

INSTRUCAO_TUTOR = (
    "Você é um tutor paciente para estudantes. Responda em português do Brasil, "
    "com linguagem clara e exemplos do dia a dia. Explique passo a passo quando o "
    "assunto for complexo e termine perguntando se o aluno quer aprofundar algum ponto."
)

INSTRUCAO_REDACAO = (
    "Você ajuda estudantes a escrever redações com voz natural, como uma pessoa de verdade escreveria. "
    "Regras: português do Brasil; varie o tamanho das frases; use vocabulário do dia a dia adequado ao nível pedido; "
    "evite clichês e expressões batidas de texto gerado por IA (como 'vale ressaltar', 'em suma', 'no mundo atual', "
    "'nos dias de hoje', 'é inegável que'); traga exemplos concretos e um ponto de vista coerente; "
    "sem listas, sem títulos (a menos que peçam), sem emojis. "
    "Entregue apenas o texto da redação, sem comentários antes ou depois. "
    "Se o aluno enviar ideias ou um rascunho, aproveite-os e mantenha o jeito dele de falar."
)

INSTRUCAO_DETECTOR = (
    "Você analisa textos em português e estima se parecem escritos por IA ou por uma pessoa. "
    "Seja honesto sobre a incerteza: nenhum método é confiável, e textos formais, escolares ou de quem "
    "escreve em segunda língua podem parecer IA mesmo sendo humanos. Responda em português do Brasil, "
    "em Markdown, neste formato: "
    "**Veredito:** (Provavelmente humano, Inconclusivo ou Provavelmente IA) com uma frase de justificativa; "
    "**Sinais de texto gerado por IA:** lista curta, citando trechos curtos do próprio texto (até 6 palavras cada); "
    "**Sinais de escrita humana:** lista curta; "
    "**Confiança:** baixa, média ou alta; "
    "**Aviso:** uma linha lembrando que isso é uma estimativa e não uma prova. "
    "Nunca invente trechos que não estejam no texto."
)

INSTRUCAO_HUMANIZAR = (
    "Você revisa textos de estudantes para que soem naturais e com voz própria. Reescreva o texto recebido "
    "em português do Brasil, mantendo o sentido, os argumentos e os fatos (não invente dados, citações ou fontes). "
    "Varie o tamanho das frases, troque expressões batidas por palavras do dia a dia do nível indicado, "
    "reduza repetições e excesso de conectivos e deixe o ritmo mais natural. Mantenha aproximadamente o mesmo tamanho. "
    "Se houver um exemplo da escrita do aluno, aproxime o estilo dele. "
    "Entregue apenas o texto reescrito, sem comentários."
)

SAUDACAO = "Olá! Sou o seu assistente de estudos. Em que matéria posso ajudar hoje?"

SUGESTOES = [
    ("🌱", "Explique a fotossíntese de forma simples"),
    ("📜", "Faça um resumo sobre a Revolução Francesa"),
    ("🧮", "Crie 5 questões sobre frações com gabarito"),
]

AVATAR_IA = "⚡"
AVATAR_USUARIO = "🧑‍🎓"
MAX_MENSAGENS_ENVIADAS = 20  # limita o histórico enviado ao modelo


# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def novo_chat():
    return [{"role": "assistant", "content": SAUDACAO}]


def montar_historico(mensagens):
    """Converte o histórico do chat para o formato do Gemini."""
    historico = []
    for m in mensagens:
        if m.get("erro"):
            continue
        papel = "user" if m["role"] == "user" else "model"
        historico.append({"role": papel, "parts": [m["content"]]})
    historico = historico[-MAX_MENSAGENS_ENVIADAS:]
    while historico and historico[0]["role"] == "model":
        historico.pop(0)  # o Gemini espera começar por uma fala do usuário
    return historico


def stream_resposta(historico, nome_modelo, temperatura, ao_iniciar=None, instrucao=INSTRUCAO_TUTOR):
    """Gera a resposta palavra por palavra (efeito de digitação)."""
    model = genai.GenerativeModel(nome_modelo, system_instruction=instrucao)
    resposta = model.generate_content(
        historico,
        generation_config=genai.GenerationConfig(temperature=temperatura),
        stream=True,
    )
    primeiro = True
    for pedaco in resposta:
        try:
            texto = pedaco.text
        except ValueError:
            continue  # pedaço sem texto (ex.: bloqueado pelo filtro de segurança)
        if not texto:
            continue
        if primeiro and ao_iniciar:
            ao_iniciar()
            primeiro = False
        for parte in re.split(r"(\s+)", texto):
            if parte:
                yield parte
                time.sleep(0.012)


def erro_amigavel(e):
    msg = str(e).lower()
    if "api key" in msg or "api_key" in msg or "permission" in msg or "403" in msg:
        return "A chave da API parece inválida ou sem permissão. Confira o GEMINI_API_KEY."
    if "429" in msg or "quota" in msg or "exhausted" in msg:
        return (
            "O limite gratuito da API foi atingido. Espere um minuto e, se continuar, "
            "troque para o Gemini 3.5 Flash-Lite na barra lateral (tem mais usos grátis por dia)."
        )
    if "404" in msg or "not found" in msg:
        return "Esse modelo não está disponível. Troque o modelo na barra lateral."
    return "Não consegui falar com o Gemini agora. Tente novamente em instantes."


def conversa_em_texto(mensagens):
    linhas = []
    for m in mensagens:
        if m.get("erro"):
            continue
        quem = "Você" if m["role"] == "user" else "Tutor"
        linhas.append(f"{quem}: {m['content']}")
    return "\n\n".join(linhas)


def calcular(operacao, a, b):
    """Retorna (resultado, simbolo, erro)."""
    if operacao.startswith("Multiplicação"):
        res, simbolo = a * b, "×"
    elif operacao.startswith("Adição"):
        res, simbolo = a + b, "+"
    elif operacao.startswith("Subtração"):
        res, simbolo = a - b, "−"
    elif operacao.startswith("Divisão"):
        if b == 0:
            return None, "÷", "Não é possível dividir por zero."
        res, simbolo = a / b, "÷"
    elif operacao.startswith("Potência"):
        try:
            res, simbolo = a ** b, "^"
        except OverflowError:
            return None, "^", "O resultado é grande demais para calcular."
        if isinstance(res, complex):
            return None, "^", "O resultado não é um número real."
    else:
        res, simbolo = (a + b) / 2, "média"

    if not math.isfinite(res):
        return None, simbolo, "O resultado é grande demais para calcular."
    return res, simbolo, None


def formatar_numero(n):
    if float(n).is_integer():
        return str(int(n))
    return f"{n:.6g}".replace(".", ",")


# =====================================================================
# BARRA LATERAL
# =====================================================================
if "mensagens" not in st.session_state:
    st.session_state.mensagens = novo_chat()

with st.sidebar:
    st.header("Configurações da IA")
    modelo_selecionado = st.selectbox("Modelo", list(MODELOS.keys()))
    temperatura = st.slider(
        "Criatividade",
        0.0, 2.0, 1.0, 0.1,
        help="Nos modelos Gemini 3, o ideal é deixar em 1.0. Valores menores deixam as respostas mais objetivas.",
    )

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.mensagens = novo_chat()
        st.rerun()

    if len(st.session_state.mensagens) > 1:
        st.download_button(
            "Baixar conversa",
            data=conversa_em_texto(st.session_state.mensagens),
            file_name="conversa-de-estudos.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    if api_conectada:
        st.success("Conectado à API do Gemini")
    else:
        st.warning("Chave GEMINI_API_KEY não encontrada nos Secrets")
    st.caption("Central de Inteligência Artificial v3.0")

nome_modelo = MODELOS[modelo_selecionado]

# =====================================================================
# CABEÇALHO
# =====================================================================
st.markdown(
    """
<div class="hero">
  <h1><span class="bolt">⚡</span> Central Inteligente para Estudantes</h1>
  <p>Tire dúvidas, resolva contas e analise materiais de estudo com ajuda da IA.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Chat de estudos", "🧮 Cálculos e gráficos", "🔍 Analisar material", "✍️ Redação", "🧠 Mapa mental"]
)

# =====================================================================
# ABA 1: CHAT
# =====================================================================
with tab1:
    # Histórico
    for msg in st.session_state.mensagens:
        avatar = AVATAR_IA if msg["role"] == "assistant" else AVATAR_USUARIO
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Sugestões (só aparecem antes da primeira pergunta)
    if len(st.session_state.mensagens) == 1:
        st.caption("Por onde começar? Toque em uma sugestão ou escreva a sua dúvida.")
        colunas = st.columns(len(SUGESTOES))
        for i, (coluna, (emoji, texto)) in enumerate(zip(colunas, SUGESTOES)):
            if coluna.button(f"{emoji}  {texto}", use_container_width=True, key=f"sug_{i}"):
                st.session_state.mensagens.append({"role": "user", "content": texto})
                st.rerun()

    # Gera a resposta quando a última mensagem é do usuário
    if st.session_state.mensagens[-1]["role"] == "user":
        with st.chat_message("assistant", avatar=AVATAR_IA):
            if not api_conectada:
                aviso = "Adicione a chave GEMINI_API_KEY nos Secrets para eu poder responder."
                st.warning(aviso)
                st.session_state.mensagens.append({"role": "assistant", "content": aviso, "erro": True})
            else:
                pensando = st.empty()
                pensando.caption("Pensando…")
                try:
                    historico = montar_historico(st.session_state.mensagens)
                    texto = st.write_stream(
                        stream_resposta(historico, nome_modelo, temperatura, ao_iniciar=pensando.empty)
                    )
                    if not texto or not texto.strip():
                        pensando.empty()
                        texto = "Não consegui gerar uma resposta para isso. Tente reformular a pergunta."
                        st.info(texto)
                        st.session_state.mensagens.append({"role": "assistant", "content": texto, "erro": True})
                    else:
                        st.session_state.mensagens.append({"role": "assistant", "content": texto})
                except Exception as e:
                    pensando.empty()
                    aviso = erro_amigavel(e)
                    st.error(aviso)
                    st.caption(f"Detalhe técnico: {str(e)[:200]}")
                    st.session_state.mensagens.append({"role": "assistant", "content": aviso, "erro": True})

    if pergunta := st.chat_input("Digite sua dúvida (ex: 'Explique a fotossíntese de forma simples')"):
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.rerun()

# =====================================================================
# ABA 2: CÁLCULOS
# =====================================================================
with tab2:
    st.subheader("Cálculos e gráficos")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        val_a = st.number_input("Valor A", value=3.0, step=1.0)
        val_b = st.number_input("Valor B", value=3.0, step=1.0)
        operacao = st.selectbox(
            "Operação",
            ["Multiplicação (×)", "Adição (+)", "Subtração (−)", "Divisão (÷)", "Potência (A elevado a B)", "Média"],
        )

        if st.button("Calcular", type="primary"):
            res, simbolo, erro = calcular(operacao, val_a, val_b)
            st.session_state["calc"] = {"a": val_a, "b": val_b, "res": res, "simbolo": simbolo, "erro": erro}

    with col2:
        calc = st.session_state.get("calc")
        if calc is None:
            st.info("Escolha os valores e a operação e clique em Calcular.")
        elif calc["erro"]:
            st.error(calc["erro"])
        else:
            a_txt, b_txt = formatar_numero(calc["a"]), formatar_numero(calc["b"])
            if calc["simbolo"] == "média":
                conta = f"Média de {a_txt} e {b_txt}"
            else:
                conta = f"{a_txt} {calc['simbolo']} {b_txt}"
            st.metric(conta, formatar_numero(calc["res"]))
            grafico = pd.DataFrame(
                {"Valores": [calc["a"], calc["b"], calc["res"]]},
                index=["Valor A", "Valor B", "Resultado"],
            )
            st.bar_chart(grafico)

# =====================================================================
# ABA 3: ANÁLISE DE MATERIAL
# =====================================================================
with tab3:
    st.subheader("Analisar material de estudo")
    arquivo = st.file_uploader(
        "Envie uma foto de caderno, exercício ou um PDF/TXT:",
        type=["jpg", "jpeg", "png", "pdf", "txt"],
    )

    if arquivo is None:
        st.info("Envie um arquivo para começar.")
    else:
        eh_imagem = arquivo.type.startswith("image/")
        if eh_imagem:
            st.image(arquivo, caption=arquivo.name, use_container_width=True)
        else:
            st.caption(f"Arquivo carregado: {arquivo.name}")

        pedido = st.text_input(
            "O que você quer que a IA faça?",
            value="Resuma o conteúdo e liste os pontos principais.",
        )

        if st.button("Analisar com IA", type="primary"):
            if not api_conectada:
                st.warning("Adicione a chave GEMINI_API_KEY nos Secrets para usar a análise.")
            else:
                try:
                    if eh_imagem:
                        conteudo = [pedido, Image.open(arquivo)]
                    elif arquivo.type == "application/pdf":
                        conteudo = [pedido, {"mime_type": "application/pdf", "data": arquivo.getvalue()}]
                    else:
                        texto_arquivo = arquivo.getvalue().decode("utf-8", errors="ignore")
                        conteudo = [f"{pedido}\n\n---\n{texto_arquivo}"]

                    with st.spinner("Lendo o material..."):
                        model = genai.GenerativeModel(nome_modelo, system_instruction=INSTRUCAO_TUTOR)
                        resposta = model.generate_content(conteudo)
                        resultado = resposta.text
                    st.session_state["analise"] = {"arquivo": arquivo.name, "texto": resultado}
                except Exception as e:
                    st.session_state.pop("analise", None)
                    st.error(erro_amigavel(e))
                    st.caption(f"Detalhe técnico: {str(e)[:200]}")

        # O resultado fica salvo e não some quando a página recarrega
        analise = st.session_state.get("analise")
        if analise and analise["arquivo"] == arquivo.name:
            st.divider()
            st.markdown(analise["texto"])

def executar_stream(chave, pedido, instrucao, temp, msg_carregando):
    """Gera um texto com efeito de digitação e guarda o resultado no session_state."""
    st.divider()
    pensando = st.empty()
    pensando.caption(msg_carregando)
    try:
        texto = st.write_stream(
            stream_resposta(
                [{"role": "user", "parts": [pedido]}],
                nome_modelo,
                temp,
                ao_iniciar=pensando.empty,
                instrucao=instrucao,
            )
        )
        if texto and texto.strip():
            st.session_state[chave] = texto
        else:
            pensando.empty()
            st.info("Não consegui gerar o resultado. Tente de novo.")
    except Exception as e:
        pensando.empty()
        st.error(erro_amigavel(e))
        st.caption(f"Detalhe técnico: {str(e)[:200]}")


# =====================================================================
# ABA 4: REDAÇÃO
# =====================================================================
with tab4:
    sub_escrever, sub_detectar, sub_humanizar = st.tabs(
        ["✍️ Escrever", "🕵️ Detectar IA", "🧑 Humanizar texto"]
    )

    with sub_escrever:
        st.subheader("Escrever redação")
        st.caption("Texto com jeito de gente: frases variadas, exemplos reais e sem aquelas expressões batidas.")

        tema = st.text_input("Tema da redação", placeholder="Ex.: Os impactos das redes sociais na saúde mental dos jovens")

        c1, c2 = st.columns(2)
        with c1:
            tipo = st.selectbox(
                "Tipo de texto",
                [
                    "Dissertativo-argumentativo (modelo ENEM)",
                    "Opinião / artigo",
                    "Narrativa",
                    "Carta",
                    "Resumo ou resenha",
                ],
            )
            nivel = st.selectbox("Nível", ["Ensino fundamental", "Ensino médio", "Ensino superior"], index=1)
        with c2:
            tom = st.selectbox("Tom", ["Natural", "Formal", "Descontraído"])
            tamanho = st.slider("Tamanho aproximado (palavras)", 100, 800, 300, 50)

        ideias = st.text_area(
            "Suas ideias ou rascunho (opcional)",
            placeholder="Cole aqui pontos que você quer usar ou um rascunho. A IA aproveita e mantém o seu jeito de escrever.",
            height=120,
        )

        if st.button("Escrever redação", type="primary"):
            if not tema.strip():
                st.warning("Escreva o tema da redação primeiro.")
            elif not api_conectada:
                st.warning("Adicione a chave GEMINI_API_KEY nos Secrets para usar este recurso.")
            else:
                pedido = (
                    f"Escreva um texto do tipo: {tipo}.\n"
                    f"Tema: {tema.strip()}\n"
                    f"Nível: {nivel}. Tom: {tom}. Tamanho: cerca de {tamanho} palavras."
                )
                if tipo.startswith("Dissertativo"):
                    pedido += (
                        "\nEstrutura: introdução com tese, dois parágrafos de desenvolvimento com argumentos e "
                        "repertório, e conclusão com proposta de intervenção (agente, ação, meio, finalidade e detalhamento)."
                    )
                if ideias.strip():
                    pedido += f"\n\nIdeias/rascunho do aluno para aproveitar:\n{ideias.strip()}"

                st.divider()
                pensando = st.empty()
                pensando.caption("Escrevendo…")
                try:
                    texto = st.write_stream(
                        stream_resposta(
                            [{"role": "user", "parts": [pedido]}],
                            nome_modelo,
                            max(temperatura, 0.7),
                            ao_iniciar=pensando.empty,
                            instrucao=INSTRUCAO_REDACAO,
                        )
                    )
                    if texto and texto.strip():
                        st.session_state["redacao"] = {"tema": tema.strip(), "texto": texto}
                    else:
                        pensando.empty()
                        st.info("Não consegui gerar o texto. Tente mudar o tema ou reformular.")
                except Exception as e:
                    pensando.empty()
                    st.error(erro_amigavel(e))
                    st.caption(f"Detalhe técnico: {str(e)[:200]}")
        else:
            salva = st.session_state.get("redacao")
            if salva:
                st.divider()
                st.markdown(salva["texto"])

        salva = st.session_state.get("redacao")
        if salva:
            st.caption(f"{len(salva['texto'].split())} palavras")
            st.download_button(
                "Baixar redação",
                data=salva["texto"],
                file_name="redacao.txt",
                mime="text/plain",
            )

    # -----------------------------------------------------------------
    # Detectar se o texto parece escrito por IA
    # -----------------------------------------------------------------
    with sub_detectar:
        st.subheader("Esse texto parece escrito por IA?")
        st.caption(
            "É uma estimativa baseada em padrões do texto. Não é prova e pode errar, "
            "principalmente em textos formais ou de quem escreve em outra língua."
        )
        texto_det = st.text_area("Cole o texto", height=220, key="texto_det")

        if st.button("Analisar texto", type="primary", key="btn_det"):
            if len(texto_det.split()) < 40:
                st.warning("Cole um texto com pelo menos 40 palavras para a análise fazer sentido.")
            elif not api_conectada:
                st.warning("Adicione a chave GEMINI_API_KEY nos Secrets para usar este recurso.")
            else:
                executar_stream(
                    "deteccao",
                    f"Analise o texto abaixo:\n\n---\n{texto_det.strip()}\n---",
                    INSTRUCAO_DETECTOR,
                    temperatura,
                    "Analisando…",
                )
        elif st.session_state.get("deteccao"):
            st.divider()
            st.markdown(st.session_state["deteccao"])

    # -----------------------------------------------------------------
    # Humanizar: deixar o texto mais natural, com a voz do aluno
    # -----------------------------------------------------------------
    with sub_humanizar:
        st.subheader("Deixar o texto mais natural")
        st.caption(
            "Reescreve mantendo as suas ideias, com frases mais variadas e menos expressões batidas. "
            "Leia o resultado e ajuste com as suas palavras. Isso não garante que um detector vá aprovar o texto, "
            "e vale conferir as regras da sua escola sobre uso de IA."
        )
        texto_hum = st.text_area("Cole o texto que quer revisar", height=220, key="texto_hum")
        nivel_hum = st.selectbox(
            "Nível de linguagem",
            ["Ensino fundamental", "Ensino médio", "Ensino superior"],
            index=1,
            key="nivel_hum",
        )
        amostra = st.text_area(
            "Um trecho que você mesmo escreveu (opcional)",
            placeholder="Se colar um texto seu, a IA tenta se aproximar do seu jeito de escrever.",
            height=100,
            key="amostra_hum",
        )

        if st.button("Humanizar texto", type="primary", key="btn_hum"):
            if len(texto_hum.split()) < 30:
                st.warning("Cole um texto com pelo menos 30 palavras.")
            elif not api_conectada:
                st.warning("Adicione a chave GEMINI_API_KEY nos Secrets para usar este recurso.")
            else:
                pedido_hum = f"Nível de linguagem: {nivel_hum}.\n\nTexto para reescrever:\n{texto_hum.strip()}"
                if amostra.strip():
                    pedido_hum += f"\n\nExemplo da escrita do aluno (imite o estilo):\n{amostra.strip()}"
                executar_stream(
                    "humanizado",
                    pedido_hum,
                    INSTRUCAO_HUMANIZAR,
                    max(temperatura, 0.7),
                    "Reescrevendo…",
                )
        elif st.session_state.get("humanizado"):
            st.divider()
            st.markdown(st.session_state["humanizado"])

        resultado_hum = st.session_state.get("humanizado")
        if resultado_hum:
            st.caption(f"{len(resultado_hum.split())} palavras")
            st.download_button(
                "Baixar texto",
                data=resultado_hum,
                file_name="texto-humanizado.txt",
                mime="text/plain",
                key="dl_hum",
            )


# =====================================================================
# ABA 5: MAPA MENTAL
# =====================================================================
INSTRUCAO_MAPA = (
    "Você cria mapas mentais para estudantes, em português do Brasil. "
    "Responda APENAS com JSON válido, sem texto antes ou depois e sem blocos de código, neste formato: "
    '{"nome": "tema central", "filhos": [{"nome": "ramo", "filhos": [{"nome": "subitem", "filhos": []}]}]}. '
    "Cada nome deve ser curto (até 6 palavras). Organize do geral para o específico, sem repetir ideias. "
    "Use no máximo 3 níveis abaixo do tema central. Não invente dados: se houver um texto base, use só o conteúdo dele."
)

CORES_RAMOS = ["#4C9AFF", "#3DD6A3", "#FFB020", "#FF6B6B", "#A78BFA", "#F472B6", "#2DD4BF", "#FB923C"]
LIMITE_FILHOS = [8, 5, 4]  # máximo de filhos por nível
LIMITE_NOS = 80


def limpar_no(no, nivel=0, contador=None):
    """Valida e limita a estrutura devolvida pela IA."""
    if contador is None:
        contador = [0]
    contador[0] += 1
    if isinstance(no, str):
        no = {"nome": no, "filhos": []}
    if not isinstance(no, dict):
        no = {"nome": "", "filhos": []}
    nome = " ".join(str(no.get("nome", "")).split())[:80] or "(sem título)"
    filhos = []
    if nivel < len(LIMITE_FILHOS):
        for f in (no.get("filhos") or [])[: LIMITE_FILHOS[nivel]]:
            if contador[0] >= LIMITE_NOS:
                break
            filhos.append(limpar_no(f, nivel + 1, contador))
    return {"nome": nome, "filhos": filhos}


def extrair_json(texto):
    texto = texto.strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto).strip()
    return json.loads(texto)


def gerar_mapa(tema, ramos, detalhe, texto_base):
    detalhes = {
        "Resumido": "até 2 subitens por ramo",
        "Médio": "até 3 subitens por ramo",
        "Detalhado": "até 4 subitens por ramo, com detalhes de terceiro nível quando fizer sentido",
    }
    pedido = f"Tema central: {tema}\nRamos principais: {ramos}\nDetalhe: {detalhes[detalhe]}."
    if texto_base.strip():
        pedido += f"\n\nTexto base para organizar:\n{texto_base.strip()}"
    model = genai.GenerativeModel(nome_modelo, system_instruction=INSTRUCAO_MAPA)
    resposta = model.generate_content(
        pedido,
        generation_config=genai.GenerationConfig(
            temperature=temperatura, response_mime_type="application/json"
        ),
    )
    dados = extrair_json(resposta.text)
    if isinstance(dados, list):
        dados = {"nome": tema, "filhos": dados}
    raiz = limpar_no(dados)
    if not raiz["filhos"]:
        raise ValueError("mapa vazio")
    return raiz


def mapa_para_markdown(no, nivel=0):
    if nivel == 0:
        linhas = [f"# {no['nome']}", ""]
    else:
        linhas = [f"{'  ' * (nivel - 1)}- {no['nome']}"]
    for f in no["filhos"]:
        linhas += mapa_para_markdown(f, nivel + 1)
    return linhas


def mapa_para_dot(raiz):
    def rotulo(texto):
        texto = texto.replace("\\", " ").replace('"', "'")
        return "\\n".join(textwrap.wrap(texto, 24)) or texto

    linhas = [
        "digraph G {",
        "rankdir=LR;",
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, margin="0.18,0.10"];',
        'edge [color="#5A6A82", arrowhead=none];',
        f'n0 [label="{rotulo(raiz["nome"])}", fillcolor="#FFB020", color="#FFB020", fontcolor="#14110A", fontsize=15];',
    ]
    contador = [0]

    def adicionar(no, id_pai, nivel, cor):
        for i, filho in enumerate(no["filhos"]):
            contador[0] += 1
            id_no = f"n{contador[0]}"
            cor_no = CORES_RAMOS[i % len(CORES_RAMOS)] if nivel == 1 else cor
            if nivel == 1:
                estilo = f'fillcolor="{cor_no}", color="{cor_no}", fontcolor="#0B0F14"'
            else:
                estilo = f'fillcolor="#161C26", color="{cor_no}", fontcolor="#E6EAF0"'
            linhas.append(f'{id_no} [label="{rotulo(filho["nome"])}", {estilo}];')
            linhas.append(f'{id_pai} -> {id_no} [color="{cor_no}"];')
            adicionar(filho, id_no, nivel + 1, cor_no)

    adicionar(raiz, "n0", 1, "#5A6A82")
    linhas.append("}")
    return "\n".join(linhas)


with tab5:
    st.subheader("Mapa mental")
    st.caption("Digite um tema, ou cole um texto, e a IA organiza as ideias em um esquema visual.")

    tema_mapa = st.text_input("Tema", placeholder="Ex.: Revolução Francesa", key="tema_mapa")

    m1, m2 = st.columns(2)
    with m1:
        ramos_mapa = st.slider("Ramos principais", 3, 8, 5, key="ramos_mapa")
    with m2:
        detalhe_mapa = st.selectbox("Nível de detalhe", ["Resumido", "Médio", "Detalhado"], index=1, key="detalhe_mapa")

    texto_mapa = st.text_area(
        "Texto base (opcional)",
        placeholder="Cole um resumo, um trecho do livro ou suas anotações. A IA organiza só o que estiver nele.",
        height=120,
        key="texto_mapa",
    )

    if st.button("Gerar mapa mental", type="primary", key="btn_mapa"):
        if not tema_mapa.strip() and not texto_mapa.strip():
            st.warning("Digite um tema ou cole um texto primeiro.")
        elif not api_conectada:
            st.warning("Adicione a chave GEMINI_API_KEY nos Secrets para usar este recurso.")
        else:
            titulo = tema_mapa.strip() or "Tema do texto"
            try:
                with st.spinner("Montando o mapa…"):
                    raiz = gerar_mapa(titulo, ramos_mapa, detalhe_mapa, texto_mapa)
                st.session_state["mapa"] = {"tema": titulo, "raiz": raiz}
            except (json.JSONDecodeError, ValueError):
                st.session_state.pop("mapa", None)
                st.error("A IA devolveu o mapa num formato inesperado. Clique em gerar de novo.")
            except Exception as e:
                st.session_state.pop("mapa", None)
                st.error(erro_amigavel(e))
                st.caption(f"Detalhe técnico: {str(e)[:200]}")

    mapa = st.session_state.get("mapa")
    if mapa:
        st.divider()
        dot = mapa_para_dot(mapa["raiz"])
        try:
            st.graphviz_chart(dot, use_container_width=True)
        except TypeError:
            st.graphviz_chart(dot)

        esquema = "\n".join(mapa_para_markdown(mapa["raiz"]))
        with st.expander("Ver em formato de texto"):
            st.markdown(esquema)
        st.download_button(
            "Baixar esquema (.md)",
            data=esquema,
            file_name="mapa-mental.md",
            mime="text/markdown",
            key="dl_mapa",
        )
