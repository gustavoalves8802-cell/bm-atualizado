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

def normalizar_texto(texto):
    """Remove acentos e deixa em minúsculo para busca segura de colunas"""
    if not isinstance(texto, str): return str(texto)
    nksel = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nksel if not unicodedata.combining(c)]).lower().strip()

@st.cache_data
def load_mapa():
    try:
        # Tenta detectar o separador
        try:
            df = pd.read_csv('MAPA-CRAFT.csv', sep=';', dtype=str)
            if len(df.columns) < 3: raise Exception()
        except:
            df = pd.read_csv('MAPA-CRAFT.csv', sep=',', dtype=str)
        
        # Normaliza nomes de colunas para evitar KeyError
        # "Categoria (preencha...)" vira "categoria (preencha...)"
        df.columns = [c.strip() for c in df.columns]
        mapeamento_colunas = {c: normalizar_texto(c) for c in df.columns}
        df = df.rename(columns=mapeamento_colunas)
        
        df['ordem_original'] = range(len(df))
        
        # Conversão numérica segura
        cols_num = ['leather', 'cloth', 'metalbar', 'planks', 'porcentagem diario']
        for col in cols_num:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '.').fillna('0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Erro ao carrergar CSV: {e}")
        return pd.DataFrame()

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

# --- PROCESSAMENTO ---
df_mapa = load_mapa()

# Nomes normalizados para lógica interna
COL_CAT = normalizar_texto('Categoria (preencha: Arma/Armadura/etc)')
COL_LIVRO = normalizar_texto('Livro')

if df_mapa.empty:
    st.error("Erro: O arquivo CSV está vazio ou não foi encontrado.")
    st.stop()

with st.sidebar:
    st.header("⚙️ CONFIGURAÇÕES")
    premium = st.checkbox("Possui Premium", value=True)
    rrr = st.number_input("Taxa de Retorno (RRR %)", value=15.2) / 100
    btn_sync = st.button("🚀 ATUALIZAR PREÇOS", use_container_width=True)

tab_bm, tab_craft, tab_logs = st.tabs(["🏛️ BLACK MARKET", "⚒️ CRAFT SYSTEM", "⚙️ LOGS"])

if btn_sync or 'dados' in st.session_state:
    if btn_sync:
        with st.spinner('Sincronizando com Albion Data Project...'):
            log_ph = tab_logs.empty()
            is_c = df_mapa[COL_CAT].str.contains('craft', na=False)
            ids_c = df_mapa[is_c]['item_id'].unique().tolist()
            ids_e = df_mapa[~is_c]['item_id'].unique().tolist()
            res = fetch_api(ids_c, "Lymhurst", 1, log_ph) + fetch_api(ids_e, "BlackMarket", 2, log_ph)
            df_res = pd.merge(pd.DataFrame(res), df_mapa, on='item_id', how='left')
            df_res['Foto'] = df_res['item_id'].apply(lambda x: f"https://render.albiononline.com/v1/item/{x}.png")
            df_res['Tier'], df_res['Enc'] = zip(*df_res['item_id'].apply(get_tier_enchant))
            df_res['t_m'] = df_res['sell_price_min_date'].apply(get_time_metrics)
            df_res['Update'], df_res['sort_time'], df_res['cor'] = zip(*df_res['t_m'])
            st.session_state['dados'] = df_res

    if 'dados' in st.session_state:
        df = st.session_state['dados'].copy()
        m_taxa = 0.935 if premium else 0.895
        dict_p = pd.Series(df.sell_price_min.values, index=df.item_id).to_dict()

        with tab_bm:
            df_bm = df[~df[COL_CAT].str.contains('craft', na=False)].copy()
            df_bm['Líquido'] = (df_bm['sell_price_min'] * m_taxa).astype(int)
            c1, c2, c3 = st.columns([1,1,2])
            f_t = c1.multiselect("Filtrar Tier BM", sorted(df_bm['Tier'].unique()), default=df_bm['Tier'].unique())
            f_e = c2.multiselect("Filtrar Encanto BM", sorted(df_bm['Enc'].unique()), default=df_bm['Enc'].unique())
            f_s = c3.text_input("🔍 Buscar Item no BM")
            mask = df_bm['Tier'].isin(f_t) & df_bm['Enc'].isin(f_e)
            if f_s: mask &= df_bm['item_name'].str.contains(f_s, case=False)
            st.dataframe(df_bm[mask].sort_values('ordem_original')[['Foto', 'item_name', 'sell_price_min', 'Líquido', 'buy_price_max', 'Update', 'cor']].style.apply(lambda x: [f'color: {x.cor}; font-weight: bold' if i==5 else '' for i in range(len(x))], axis=1).hide(['cor'], axis=1), column_config={"Foto": st.column_config.ImageColumn(""), "sell_price_min": "Preço BM", "buy_price_max": "Pedido BM"}, use_container_width=True, hide_index=True, height=600)

        with tab_craft:
            st.subheader("📦 Materiais e Diários (Lymhurst)")
            df_mat = df[df[COL_CAT].str.contains('craft', na=False)].copy()
            
            # Filtro rigoroso de capas sem diário
            df_mat = df_mat[~((df_mat['item_id'].str.contains('CAPEITEM')) & (df_mat[COL_LIVRO].isna() | (df_mat[COL_LIVRO] == '')))]
            
            mc1, mc2, mc3 = st.columns([1,1,2])
            mf_t = mc1.multiselect("Tier Material", sorted(df_mat['Tier'].unique()), default=df_mat['Tier'].unique())
            mf_e = mc2.multiselect("Encanto Material", sorted(df_mat['Enc'].unique()), default=df_mat['Enc'].unique())
            mf_s = mc3.text_input("🔍 Buscar Material/Livro")
            
            m_mask = df_mat['Tier'].isin(mf_t) & df_mat['Enc'].isin(mf_e)
            if mf_s: m_mask &= df_mat['item_name'].str.contains(mf_s, case=False)

            ed_mat = st.data_editor(df_mat[m_mask].sort_values('ordem_original')[['Foto', 'item_name', 'sell_price_min', 'Update', 'cor', 'item_id']], column_config={"sell_price_min": st.column_config.NumberColumn("Preço Lym", format="%d"), "Foto": st.column_config.ImageColumn(""), "item_id": None, "cor": None}, hide_index=True, use_container_width=True, key="ed_mat_v4")
            for i, row in ed_mat.iterrows(): dict_p[row['item_id']] = row['sell_price_min']

            st.divider()
            st.subheader("⚒️ Análise de Produção")

            def calc_final(row):
                t, e = row['Tier'], row['Enc']
                suffix = f"_LEVEL{e}@{e}" if e > 0 else ""
                pendencias = []
                
                custo_mats = 0
                for m_nome, m_id in [('leather','LEATHER'), ('cloth','CLOTH'), ('metalbar','METALBAR'), ('planks','PLANKS')]:
                    qtd = row[m_nome]
                    if qtd > 0:
                        id_completo = f"T{t}_{m_id}{suffix}"
                        preco = dict_p.get(id_completo, 0)
                        if preco <= 0: pendencias.append(m_id)
                        custo_mats += (qtd * preco)
                
                custo_rrr = custo_mats * (1 - rrr)
                lucro_livro = 0
                if row[COL_LIVRO] in MAP_LIVROS:
                    l_tipo = MAP_LIVROS[row[COL_LIVRO]]
                    p_vazio = dict_p.get(f"T{t}_JOURNAL_{l_tipo}_EMPTY", 0)
                    p_cheio = dict_p.get(f"T{t}_JOURNAL_{l_tipo}_FULL", 0)
                    if p_vazio <= 0: pendencias.append("L.Vazio")
                    if p_cheio <= 0: pendencias.append("L.Cheio")
                    lucro_livro = (p_cheio - p_vazio) * row['porcentagem diario']
                
                if row['sell_price_min'] <= 0: pendencias.append("Preço BM")
                
                custo_real = int(custo_rrr - lucro_livro)
                venda_liq = int(row['sell_price_min'] * m_taxa)
                return pd.Series([custo_real, venda_liq, venda_liq - custo_real, len(pendencias) > 0, ", ".join(pendencias)])

            df_prod = df[~df[COL_CAT].str.contains('craft', na=False)].copy()
            df_prod[['Custo Real', 'Venda Líq', 'Lucro Final', 'Erro', 'Pendência']] = df_prod.apply(calc_final, axis=1)
            
            pc1, pc2, pc3, pc4 = st.columns([1,1,1,1])
            pf_t = pc1.multiselect("Tiers Item", sorted(df_prod['Tier'].unique()), default=df_prod['Tier'].unique())
            pf_e = pc2.multiselect("Encantos Item", sorted(df_prod['Enc'].unique()), default=df_prod['Enc'].unique())
            pf_err = pc3.selectbox("Status de Dados", ["Todos", "Apenas sem Erros", "Apenas com Erros"])
            pf_s = pc4.text_input("🔍 Filtrar Equipamento")
            
            p_mask = df_prod['Tier'].isin(pf_t) & df_prod['Enc'].isin(pf_e)
            if pf_err == "Apenas sem Erros": p_mask &= ~df_prod['Erro']
            elif pf_err == "Apenas com Erros": p_mask &= df_prod['Erro']
            if pf_s: p_mask &= df_prod['item_name'].str.contains(pf_s, case=False)

            st.dataframe(
                df_prod[p_mask].sort_values('ordem_original')[['Foto', 'item_name', 'Custo Real', 'Venda Líq', 'Lucro Final', 'Pendência', 'Erro']]
                .style.apply(lambda r: ['background-color: #5e0000; color: white' if r.Erro else '' for _ in r], axis=1).hide(['Erro'], axis=1),
                column_config={"Foto": st.column_config.ImageColumn(""), "Pendência": st.column_config.TextColumn("⚠️ Atenção")},
                use_container_width=True, hide_index=True, height=600
            )

with tab_logs:
    st.write("Colunas Normalizadas:", list(df_mapa.columns))
