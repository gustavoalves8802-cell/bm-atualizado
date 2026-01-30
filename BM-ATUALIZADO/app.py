import streamlit as st
import pandas as pd
import requests
import time
import re
import unicodedata
from datetime import datetime, timezone

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Albion Analytics PRO", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #00f2ff; }
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] { border: 1px solid #30363d; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

MAP_LIVROS = {
    "Tinker's": "TOOLMAKER",
    "Blacksmith": "WARRIOR",
    "Fletcher": "HUNTER",
    "Imbuer": "MAGE"
}

def normalizar(t):
    return "".join(c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)).lower().strip()

@st.cache_data
def load_mapa():
    try:
        # Tenta carregar com ; e se falhar tenta ,
        try:
            df = pd.read_csv('MAPA-CRAFT.csv', sep=';', dtype=str)
            if len(df.columns) < 2: raise Exception()
        except:
            df = pd.read_csv('MAPA-CRAFT.csv', sep=',', dtype=str)
        
        # Limpa espaços e normaliza nomes das colunas
        df.columns = [c.strip() for c in df.columns]
        df['ordem_original'] = range(len(df))
        
        # Converte colunas numéricas (trata vírgula e ponto)
        for col in ['LEATHER', 'CLOTH', 'METALBAR', 'PLANKS', 'Porcentagem Diario']:
            target = next((c for c in df.columns if normalizar(c) == normalizar(col)), None)
            if target:
                df[target] = df[target].str.replace(',', '.').fillna('0')
                df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return pd.DataFrame()

# --- CARREGAMENTO ---
df_mapa = load_mapa()

if not df_mapa.empty:
    # LOCALIZAÇÃO FLEXÍVEL DAS COLUNAS (Evita KeyError)
    # Procura a coluna que contém 'categoria', 'livro', etc.
    col_cat = next((c for c in df_mapa.columns if 'categoria' in normalizar(c)), None)
    col_livro = next((c for c in df_mapa.columns if normalizar(c) == 'livro'), None)
    col_pct = next((c for c in df_mapa.columns if 'porcentagem' in normalizar(c)), None)

    if not col_cat:
        st.error("Coluna 'Categoria' não encontrada no CSV!")
        st.stop()
else:
    st.stop()

def get_tier_enchant(item_id):
    t = re.search(r'T(\d)', str(item_id))
    e = re.search(r'@(\d)', str(item_id))
    return int(t.group(1)) if t else 0, int(e.group(1)) if e else 0

def get_time_metrics(date_str):
    if not date_str or "0001" in str(date_str): return "-", 999999, "#888"
    try:
        dt = datetime.strptime(str(date_str).replace('Z', '').split('.')[0], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
        diff = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        if diff < 60: return f"{int(diff)}m", int(diff), "#00ff00"
        if diff < 720: return f"{int(diff//60)}h", int(diff), "#ffaa00"
        return f"{int(diff//1440)}d", int(diff), "#ff4b4b"
    except: return "-", 999999, "#888"

def fetch_api(ids, loc, qual, log_ph):
    res = []
    for i in range(0, len(ids), 100):
        batch = ids[i:i+100]
        log_ph.text(f"📡 {loc}: {min(i+100, len(ids))}/{len(ids)}")
        try:
            r = requests.get(f"https://www.albion-online-data.com/api/v2/stats/prices/{','.join(batch)}?locations={loc}&qualities={qual}", timeout=10)
            if r.status_code == 200: res.extend(r.json())
        except: pass
        time.sleep(0.1)
    return res

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("⚙️ CONFIG")
    premium = st.checkbox("Possui Premium", value=True)
    rrr = st.number_input("Taxa de Retorno (RRR %)", value=15.2) / 100
    btn_sync = st.button("🚀 ATUALIZAR PREÇOS", use_container_width=True)

tab_bm, tab_craft, tab_logs = st.tabs(["🏛️ BLACK MARKET", "⚒️ CRAFT SYSTEM", "⚙️ LOGS"])

if btn_sync or 'dados' in st.session_state:
    if btn_sync:
        with st.spinner('Sincronizando...'):
            log_ph = tab_logs.empty()
            is_c = df_mapa[col_cat].str.lower().str.contains('craft', na=False)
            ids_c = df_mapa[is_c]['item_id'].unique().tolist()
            ids_e = df_mapa[~is_c]['item_id'].unique().tolist()
            res = fetch_api(ids_c, "Lymhurst", 1, log_ph) + fetch_api(ids_e, "BlackMarket", 2, log_ph)
            df_res = pd.merge(pd.DataFrame(res), df_mapa, on='item_id', how='left')
            df_res['Foto'] = df_res['item_id'].apply(lambda x: f"https://render.albiononline.com/v1/item/{x}.png")
            df_res['Tier'], df_res['Enc'] = zip(*df_res['item_id'].apply(get_tier_enchant))
            df_res['Update'], _, df_res['cor'] = zip(*df_res['sell_price_min_date'].apply(get_time_metrics))
            st.session_state['dados'] = df_res

    if 'dados' in st.session_state:
        df = st.session_state['dados'].copy()
        m_taxa = 0.935 if premium else 0.895
        dict_p = pd.Series(df.sell_price_min.values, index=df.item_id).to_dict()

        # --- BLACK MARKET ---
        with tab_bm:
            df_bm = df[~df[col_cat].str.lower().str.contains('craft', na=False)].copy()
            df_bm['Líquido'] = (df_bm['sell_price_min'] * m_taxa).astype(int)
            st.dataframe(df_bm.sort_values('ordem_original')[['Foto', 'item_name', 'sell_price_min', 'Líquido', 'Update', 'cor']].style.apply(lambda x: [f'color: {x.cor}' if i==4 else '' for i in range(len(x))], axis=1).hide(['cor'], axis=1), column_config={"Foto": st.column_config.ImageColumn("")}, use_container_width=True, hide_index=True)

        # --- CRAFT SYSTEM ---
        with tab_craft:
            st.subheader("📦 Materiais e Diários (Lymhurst)")
            df_mat = df[df[col_cat].str.lower().str.contains('craft', na=False)].copy()
            # Filtro de capas
            df_mat = df_mat[~((df_mat['item_id'].str.contains('CAPEITEM')) & (df_mat[col_livro].isna()))]
            
            ed_mat = st.data_editor(df_mat.sort_values('ordem_original')[['Foto', 'item_name', 'sell_price_min', 'Update', 'item_id']], column_config={"sell_price_min": "Preço Lym", "Foto": st.column_config.ImageColumn(""), "item_id": None}, hide_index=True, use_container_width=True)
            for i, row in ed_mat.iterrows(): dict_p[row['item_id']] = row['sell_price_min']

            st.divider()
            st.subheader("⚒️ Análise de Lucro")

            def calc_final(row):
                t, e = row['Tier'], row['Enc']
                suffix = f"_LEVEL{e}@{e}" if e > 0 else ""
                falha = []
                
                c_mat = 0
                for m_id in ['LEATHER', 'CLOTH', 'METALBAR', 'PLANKS']:
                    target_col = next((c for c in df_mapa.columns if normalizar(c) == normalizar(m_id)), None)
                    if target_col:
                        qtd = row[target_col]
                        if qtd > 0:
                            p = dict_p.get(f"T{t}_{m_id}{suffix}", 0)
                            if p <= 0: falha.append(m_id)
                            c_mat += (qtd * p)
                
                c_rrr = c_mat * (1 - rrr)
                l_livro = 0
                if row[col_livro] in MAP_LIVROS:
                    tipo = MAP_LIVROS[row[col_livro]]
                    pv = dict_p.get(f"T{t}_JOURNAL_{tipo}_EMPTY", 0)
                    pc = dict_p.get(f"T{t}_JOURNAL_{tipo}_FULL", 0)
                    if pv <= 0 or pc <= 0: falha.append("Livro")
                    l_livro = (pc - pv) * row[col_pct]
                
                if row['sell_price_min'] <= 0: falha.append("Preço BM")
                
                custo_final = int(c_rrr - l_livro)
                venda_liq = int(row['sell_price_min'] * m_taxa)
                return pd.Series([custo_final, venda_liq, venda_liq - custo_final, len(falha) > 0, ", ".join(falha)])

            df_prod = df[~df[col_cat].str.lower().str.contains('craft', na=False)].copy()
            df_prod[['Custo Real', 'Venda Líq', 'Lucro', 'Erro', 'Pendente']] = df_prod.apply(calc_final, axis=1)
            
            status = st.selectbox("Filtrar Status", ["Todos", "Apenas sem Erros", "Apenas com Erros"])
            mask = df_prod['item_id'].notna()
            if status == "Apenas sem Erros": mask &= ~df_prod['Erro']
            elif status == "Apenas com Erros": mask &= df_prod['Erro']

            st.dataframe(
                df_prod[mask].sort_values('ordem_original')[['Foto', 'item_name', 'Custo Real', 'Venda Líq', 'Lucro', 'Pendente', 'Erro']]
                .style.apply(lambda r: ['background-color: #5e0000' if r.Erro else '' for _ in r], axis=1).hide(['Erro'], axis=1),
                column_config={"Foto": st.column_config.ImageColumn("")}, use_container_width=True, hide_index=True
            )
