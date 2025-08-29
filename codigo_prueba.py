
from pathlib import Path
import shutil

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "modelos"

from pathlib import Path

def _file_state(p: Path) -> tuple[int, int]:
    """Clave de caché basada en (tamaño, mtime). Si no existe: (0,0)."""
    try:
        s = p.stat()
        return (int(s.st_size), int(s.st_mtime_ns))
    except Exception:
        return (0, 0)

legacy_map = {
    # Excels
    "turnos_preprocesado.xlsx": DATA / "turnos_preprocesado.xlsx",
    "tanques_preprocesado.xlsx": DATA / "tanques_preprocesado.xlsx",
    "Capacidades tanques.xlsx": DATA / "Capacidades tanques.xlsx",
    "inventario_actual.xlsx": DATA / "inventario_actual.xlsx",          
    "aforos_unificado.xlsx": DATA / "aforos_unificado.xlsx",                               

    # Modelos
    "modelo_predictivo_turnos_reentrenado.pkl": MODELS / "modelo_predictivo_turnos_reentrenado.pkl",
    "modelo_predictivo_tanques_reentrenado.pkl": MODELS / "modelo_predictivo_tanques_reentrenado.pkl",
}

for legacy_name, real_path in legacy_map.items():
    legacy_path = ROOT / legacy_name
    try:
        if not legacy_path.exists() and real_path.exists():
            
            shutil.copy2(real_path, legacy_path)
    except Exception:
        
        pass


import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from datetime import timedelta, datetime, timezone
from pathlib import Path


st.set_page_config(
    page_title="Predicción de Combustible - EDS ARAUCA",
    page_icon="static/logo-eds-arauca.jpg",   # usa tu logo como ícono
    layout="wide"
)

# 🔗 Inyectar manifest, iconos y color de tema
st.markdown("""
<link rel="manifest" href="/static/manifest.webmanifest" type="application/manifest+json">
<link rel="apple-touch-icon" href="/static/icon-192.png">
<meta name="theme-color" content="#FF6600">
""", unsafe_allow_html=True)

# Encabezado con logo + título
col_logo, col_title = st.columns([1, 12])
with col_logo:
    st.image("static/logo-eds-arauca.jpg", width=100)  # ajusta el tamaño si quieres
with col_title:
    st.markdown(
        "<h1 style='margin:0;'>Sistema Inteligente de Predicción y Logística de Combustible EDS ARAUCA</h1>",
        unsafe_allow_html=True
    )



import streamlit as st
from time import time

def _get_users_from_secrets():
    """
    Espera en st.secrets algo así:
    [auth]
    usuarios = "sebas:Sebitas12, liliana:lili123*, Arauca:Arauca123*"
    # (opcional) minutos de inactividad para cerrar sesión
    timeout_minutes = 60
    """
    cfg = st.secrets.get("auth", {})
    raw = cfg.get("usuarios", "")
    pares = [p.strip() for p in raw.split(",") if p.strip()]
    users = {}
    for par in pares:
        if ":" in par:
            u, p = par.split(":", 1)
            users[u.strip()] = p.strip()
    timeout = int(cfg.get("timeout_minutes", 60))
    return users, timeout

def require_basic_login():
    users, timeout_minutes = _get_users_from_secrets()
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state._last_touch = time()

    # auto-logout por inactividad
    if st.session_state.logged_in:
        if time() - st.session_state._last_touch > timeout_minutes * 60:
            st.session_state.clear()
            st.warning("Sesión cerrada por inactividad.")
            st.stop()
        else:
            st.session_state._last_touch = time()

    if not st.session_state.logged_in:
        st.title("Inicia Sesion")
        user = st.text_input("Usuario")
        pwd = st.text_input("Contraseña", type="password")
        col1, col2 = st.columns([1,3])
        with col1:
            if st.button("Entrar", use_container_width=True):
                if user in users and pwd == users[user]:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.session_state._last_touch = time()
                    st.rerun()
                else:
                    st.error("Credenciales inválidas")
        st.stop()

    # Barra pequeña con usuario y logout
    c1, c2, c3 = st.columns([6,3,1])
    with c2:
        st.caption(f"👤 {st.session_state.user}")
    with c3:
        if st.button("Salir"):
            st.session_state.clear()
            st.rerun()

# Llamar ANTES de la lógica de la app
require_basic_login()
# ================== FIN LOGIN SIMPLE ==================


with st.sidebar:
    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()



DATA = Path(__file__).parent / "data"

@st.cache_data
def cargar_datos_turnos(_cache_key: tuple[int, int]):
    path = DATA / "turnos_preprocesado.xlsx"
    df = pd.read_excel(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(by=["Producto", "Turno", "Fecha"]).reset_index(drop=True)
    df["Lag_1"] = df.groupby(["Producto", "Turno"])["Galones"].shift(1)
    df["MediaMovil_3"] = df.groupby(["Producto", "Turno"])["Galones"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["MediaMovil_7"] = df.groupby(["Producto", "Turno"])["Galones"].transform(lambda s: s.shift(1).rolling(7).mean())
    return df

@st.cache_data
def cargar_datos_tanques(_cache_key: tuple[int, int]):
    path = DATA / "tanques_preprocesado.xlsx"
    df = pd.read_excel(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(by=["Producto", "Tanque", "Fecha"]).reset_index(drop=True)
    df["Lag_1"] = df.groupby(["Producto", "Tanque"])["Galones"].shift(1)
    df["MediaMovil_3"] = df.groupby(["Producto", "Tanque"])["Galones"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["MediaMovil_7"] = df.groupby(["Producto", "Tanque"])["Galones"].transform(lambda s: s.shift(1).rolling(7).mean())
    return df


_turnos_key  = _file_state(DATA / "turnos_preprocesado.xlsx")
_tanques_key = _file_state(DATA / "tanques_preprocesado.xlsx")

df_turnos  = cargar_datos_turnos(_turnos_key)
df_tanques = cargar_datos_tanques(_tanques_key)

@st.cache_resource
def cargar_modelo_turnos():
    return joblib.load("modelo_predictivo_turnos_reentrenado.pkl")

@st.cache_resource
def cargar_modelo_tanques():
    return joblib.load("modelo_predictivo_tanques_reentrenado.pkl")

modelo_turnos = cargar_modelo_turnos()
modelo_tanques = cargar_modelo_tanques()


def cargar_inventario_actual():
    """
    Crea inventario_actual.csv si no existe, con tanques/producto y galones=0,
    y lo devuelve como DataFrame.
    """
    path = "inventario_actual.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Fecha"])
    base = (
        df_tanques[["Tanque","Producto"]]
        .drop_duplicates()
        .assign(Medida_cm=np.nan, Galones=0.0, Fecha=pd.to_datetime("today"))
    )
    base.to_csv(path, index=False)
    return base



def fecha_pedido_carrotanque_unica(fechas_pedido: dict):
    """
    Devuelve la fecha única sugerida de pedido (mínima entre productos),
    calculada a partir del dict {'ACPM': fecha|None, 'CORRIENTE': fecha|None, ...}.
    Si no hay fechas válidas, retorna None.
    """
    import pandas as pd
    if not isinstance(fechas_pedido, dict) or len(fechas_pedido) == 0:
        return None
    try:
        fechas_validas = [pd.to_datetime(v) for v in fechas_pedido.values() if v]
        return min(fechas_validas).date() if fechas_validas else None
    except Exception:
        return None





def detectar_producto_desde_tanque(nombre_tanque: str) -> str:
    n = str(nombre_tanque).upper()
    if "ACPM" in n: return "ACPM"
    if "CORRIENTE" in n or "CTE" in n: return "CORRIENTE"
    return "SUPREME"



def generar_pred_por_tanques(df_tanques_hist, modelo_tanques, fecha_inicio, dias):
    base = df_tanques_hist.copy().sort_values("Fecha")
    base["Tanque"] = base["Tanque"].astype(str)
    base["Producto"] = base["Producto"].astype(str)

    productos = sorted(base["Producto"].unique().tolist())
    preds = []

    for i in range(dias):
        f = pd.to_datetime(fecha_inicio).normalize() + timedelta(days=i)
        dow = f.weekday()
        for prod in productos:
            hb = base[base["Producto"] == prod]
            if hb.empty:
                continue
            
            tanques_prod = sorted(hb["Tanque"].unique().tolist())
            for tq in tanques_prod:
                ht = hb[hb["Tanque"] == tq]
                if ht.empty:
                    continue
                lag1 = ht["Galones"].iloc[-1]
                m3   = ht["Galones"].tail(3).mean()
                m7   = ht["Galones"].tail(7).mean()
                onehot = {f"Producto_{p}": int(p == prod) for p in productos}
                X = pd.DataFrame([{
                    "Año": f.year, "Mes": f.month, "Día": f.day,
                    "NumeroDiaSemana": dow,
                    "Lag_1": lag1, "MediaMovil_3": m3, "MediaMovil_7": m7,
                    **onehot
                }])
                y = float(modelo_tanques.predict(X)[0])
                preds.append({"Fecha": f, "Producto": prod, "Tanque": tq, "Predicción (galones)": round(y, 2)})

                
                base = pd.concat([base, pd.DataFrame([{
                    "Fecha": f, "Producto": prod, "Tanque": tq, "Galones": y
                }])], ignore_index=True)

    return pd.DataFrame(preds)



def stock_util_por_producto(df_inv_actual, minimos_por_tanque, buffer_tanque):
    """
    Stock útil = sum_tanques max( Galones - mínimo_tanque - buffer_tanque, 0 )
    Retorna dict {producto: stock_util_total}
    """
    su_prod = {}
    for _, r in df_inv_actual.iterrows():
        tq = str(r["Tanque"])
        prod = str(r["Producto"])
        gal = float(r["Galones"])
        minimo = float(minimos_por_tanque.get(tq, 0.0))
        su = max(gal - minimo - float(buffer_tanque), 0.0)
        su_prod[prod] = su_prod.get(prod, 0.0) + su
    
    for p in ["ACPM","CORRIENTE","SUPREME"]:
        su_prod.setdefault(p, 0.0)
    return su_prod


def cobertura_exacta_por_producto(df_pred_tanques, su_por_producto, incluir_hoy=False):
    """
    Resta día a día el consumo del PRODUCTO (sumando sus tanques) hasta agotar el stock útil.
    Devuelve:
      - df_cov con columnas:
          Producto,
          StockUtilInicial (gal),
          Cobertura_dias (contando desde mañana si incluir_hoy=False),
          Rango_cubierto_completo,
          Fecha_agotamiento (día en que se acaba durante el día o '✔️ Cubierto')
      - fechas_pedido (dict {producto: fecha_agotamiento} o None)
    """
    hoy = pd.to_datetime("today").normalize().date()
    fecha_inicio = hoy if incluir_hoy else (hoy + pd.Timedelta(days=1)).date()

    cons_prod = (
        df_pred_tanques[df_pred_tanques["Fecha"] >= pd.to_datetime(fecha_inicio)]
        .groupby(["Fecha","Producto"])["Predicción (galones)"]
        .sum().reset_index().sort_values("Fecha")
    )

    out_rows, fechas_pedido = [], {}
    horizonte = cons_prod["Fecha"].dt.date.max() if not cons_prod.empty else fecha_inicio

    for prod in ["ACPM","CORRIENTE","SUPREME"]:
        su_ini = float(su_por_producto.get(prod, 0.0))
        su = su_ini
        dias = 0
        fecha_agot = None

        serie = cons_prod[cons_prod["Producto"] == prod]
        for _, r in serie.iterrows():
            su -= float(r["Predicción (galones)"])
            dias += 1
            if su <= 0 and fecha_agot is None:
                fecha_agot = pd.to_datetime(r["Fecha"]).date()
                break

        if fecha_agot:
            fin_completo = fecha_agot - pd.Timedelta(days=1)
            rango_txt = f"{fecha_inicio} → {fin_completo}"
            cov_txt = f"{max(dias-1,0)} días"
   
            fechas_pedido[prod] = fecha_agot   
            agot_txt = str(fecha_agot)
        else:
            fin_completo = horizonte
            rango_txt = f"{fecha_inicio} → {fin_completo}"
            dias_disp = (fin_completo - fecha_inicio).days + 1
            cov_txt = f"≥ {dias_disp} días"
            fechas_pedido[prod] = None
            agot_txt = "✔️ Cubierto en el horizonte"

        out_rows.append({
            "Producto": prod,
            "StockUtilInicial (gal)": round(su_ini, 1),
            "Cobertura_dias": cov_txt,
            "Rango_cubierto_completo": rango_txt,
            "Fecha_agotamiento": agot_txt
        })

    return pd.DataFrame(out_rows), fechas_pedido



def deficits_hasta_objetivo(
    df_pred_tanques: pd.DataFrame,
    df_inv_actual: pd.DataFrame,
    minimos_por_tanque: dict,
    buffer_tanque: float,
    dias_objetivo: int = 0,
    lead_time: int = 1,
    reserva_operativa_gal: float = 0,
):
    """
    Calcula:
      1) Tabla de déficits por TANQUE a la fecha objetivo (StockUtilProy < 0 => Déficit).
      2) Requerimiento por PRODUCTO con BANDEO CONSERVADOR.

    Bandeo conservador (por producto):
      - Se permite compensar solo la parte 'segura' de tanques positivos:
          surplus_seguro = max(SU_proy - reserva_operativa_gal, 0)
      - Requerimiento = max(0, sum(deficits) - sum(surplus_seguro))

    Horizonte:
      - Si dias_objetivo <= 0 → usa TODO el horizonte disponible en df_pred_tanques.
      - Si dias_objetivo > 0 → usa: hoy + lead_time + dias_objetivo (acotado al horizonte disponible).

    Retorna:
      - req_por_prod: dict {ACPM: gal, CORRIENTE: gal, SUPREME: gal}
      - df_def: DataFrame con columnas:
            Tanque, Producto, StockUtilInicial, ConsumoAcum, StockUtilProy, Deficit
    """
    if df_pred_tanques is None or df_pred_tanques.empty:
        return {"ACPM": 0.0, "CORRIENTE": 0.0, "SUPREME": 0.0}, pd.DataFrame(
            columns=["Tanque","Producto","StockUtilInicial","ConsumoAcum","StockUtilProy","Deficit"]
        )

    hoy = pd.to_datetime("today").normalize()
    
    pred_hasta = pd.to_datetime(df_pred_tanques["Fecha"]).max().normalize()
    
    if dias_objetivo is None:
        dias_objetivo = 0
    if lead_time is None:
        lead_time = 0
    if dias_objetivo <= 0:
        fecha_obj = pred_hasta
    else:
        fecha_obj = min(pred_hasta, hoy + pd.Timedelta(days=int(lead_time) + int(dias_objetivo)))

    
    su_rows = []
    for _, r in df_inv_actual.iterrows():
        tq   = str(r["Tanque"])
        prod = str(r["Producto"])
        gal  = float(r["Galones"])
        minimo = float(minimos_por_tanque.get(tq, 0.0))
        su_ini = max(gal - minimo - float(buffer_tanque), 0.0)
        su_rows.append([tq, prod, su_ini])
    df_su = pd.DataFrame(su_rows, columns=["Tanque","Producto","StockUtilInicial"])

    
    df_cons = (
        df_pred_tanques[df_pred_tanques["Fecha"] <= fecha_obj]
        .groupby(["Tanque","Producto"])["Predicción (galones)"].sum()
        .rename("ConsumoAcum").reset_index()
    )

    
    df_def = (
        df_su.merge(df_cons, on=["Tanque","Producto"], how="left")
             .fillna({"ConsumoAcum": 0.0})
    )
    df_def["StockUtilProy"] = df_def["StockUtilInicial"] - df_def["ConsumoAcum"]
    df_def["Deficit"]       = df_def["StockUtilProy"].apply(lambda x: -x if x < 0 else 0.0)

    
    req_por_prod = {}
    for p in ["ACPM", "CORRIENTE", "SUPREME"]:
        dfp = df_def[df_def["Producto"] == p]
        if dfp.empty:
            req_por_prod[p] = 0.0
            continue

        negativos = float(dfp[dfp["StockUtilProy"] < 0]["Deficit"].sum())
        
        positivos_seg = float(
            dfp[dfp["StockUtilProy"] > 0]["StockUtilProy"].apply(
                lambda su: max(su - float(reserva_operativa_gal), 0.0)
            ).sum()
        )
        req_por_prod[p] = max(0.0, negativos - positivos_seg)

    
    df_def = df_def.sort_values(
        ["Producto", "Deficit", "StockUtilProy"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    return req_por_prod, df_def




def plan_carrotanque_3_comp(
    req_por_prod: dict,
    df_deficits: "pd.DataFrame",
    carrotanques_caps: dict,
    placa_preferida: str | None = None,
):
    """
    Plan de carga/descarga para 3 compartimientos PRIORITIZANDO DÉFICIT POR PRODUCTO.
    - Asegura mínimo un compartimiento por producto con déficit (si hay espacio).
    - En la primera ronda no repite tanque si hay otros en déficit.
    - Luego completa por mayor déficit o rotación.
    """
    import pandas as pd

    if df_deficits is None or len(df_deficits) == 0:
        return pd.DataFrame(columns=["Placa","Compartimiento","Producto","Tanque","Galones asignados"])

    # Selección de placa y capacidades
    if placa_preferida and placa_preferida in carrotanques_caps:
        placa = placa_preferida
    else:
        placa = next(iter(carrotanques_caps.keys()))
    caps = list(carrotanques_caps[placa])

    df = df_deficits.copy()
    for c in ["Deficit","StockUtilProy"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df["Deficit"] = df["Deficit"].fillna(0.0)

    productos_con_deficit = (
        df.groupby("Producto")["Deficit"].sum().reset_index()
          .query("Deficit > 0")["Producto"].tolist()
    )
    severidad_prod = (
        df.groupby("Producto")["Deficit"].sum().reset_index()
          .sort_values("Deficit", ascending=False)
    )
    orden_productos = [p for p in severidad_prod["Producto"] if p in productos_con_deficit]

    plan_rows, tanques_servidos, productos_servidos = [], set(), set()

    def pick_top_deficit_by_product(cap, prod, avoid_repeat_tank=True):
        base = df[(df["Producto"] == prod) & (df["Deficit"] > 0)].copy()
        if avoid_repeat_tank:
            base = base[~base["Tanque"].isin(tanques_servidos)]
        if base.empty: return None
        c = base.sort_values(["Deficit","StockUtilProy"], ascending=[False, True]).iloc[0]
        return str(c["Producto"]), str(c["Tanque"]), float(min(cap, c["Deficit"]))

    def pick_top_deficit_any(cap, avoid_repeat_tank=False):
        base = df[df["Deficit"] > 0].copy()
        if avoid_repeat_tank:
            base = base[~base["Tanque"].isin(tanques_servidos)]
        if base.empty: return None
        c = base.sort_values(["Deficit","StockUtilProy"], ascending=[False, True]).iloc[0]
        return str(c["Producto"]), str(c["Tanque"]), float(min(cap, c["Deficit"]))

    def pick_by_rotation(cap):
        if df.empty: return None
        c = df.sort_values(["StockUtilProy"]).iloc[0]
        return str(c["Producto"]), str(c["Tanque"]), float(cap)

    comp_index = 1
    for cap in caps:
        choice = None
        
        restantes = [p for p in orden_productos if p not in productos_servidos]
        if restantes:
            for prod in restantes:
                choice = pick_top_deficit_by_product(cap, prod, avoid_repeat_tank=True)
                if choice: break
        
        if not choice: choice = pick_top_deficit_any(cap, avoid_repeat_tank=True)
        
        if not choice: choice = pick_top_deficit_any(cap, avoid_repeat_tank=False)
        
        if not choice: choice = pick_by_rotation(cap)

        if not choice:
            plan_rows.append([placa, comp_index, "", "", 0.0])
            comp_index += 1
            continue

        prod, tq, asignado = choice
        plan_rows.append([placa, comp_index, prod, tq, round(asignado, 2)])
        tanques_servidos.add(tq)
        if prod in productos_con_deficit: productos_servidos.add(prod)

        m = (df["Tanque"] == tq) & (df["Producto"] == prod)
        if m.any():
            df.loc[m, "Deficit"] = (df.loc[m, "Deficit"] - asignado).clip(lower=0.0)
            df.loc[m, "StockUtilProy"] = df.loc[m, "StockUtilProy"] + asignado

        comp_index += 1

    df_plan = pd.DataFrame(plan_rows, columns=["Placa","Compartimiento","Producto","Tanque","Galones asignados"])
    if not df_plan.empty:
        try:
            df_plan["Compartimiento"] = df_plan["Compartimiento"].astype(int) - 1
            df_plan["Compartimiento"] = df_plan["Compartimiento"].apply(lambda idx: f"{caps[idx]} gal" if 0 <= idx < len(caps) else idx)
        except Exception:
            pass
    return df_plan



# ===================== DASHBOARD DE BIENVENIDA =====================
import altair as alt

def kpi_chip(label, value, help_txt=None):
    col = st.container()
    col.metric(label, value)
    if help_txt: col.caption(help_txt)
    return col

def resumen_estado_actual_ui(pred_dias_default=4):
    st.subheader("🏁 Stock actual de combustible")

    # 1) Predicción LOCAL para KPIs (NO toca st.session_state["df_pred_tanques"])
    inicio = pd.to_datetime("today").normalize()
    try:
        df_pred_local = generar_pred_por_tanques(df_tanques, modelo_tanques, inicio, pred_dias_default)
    except Exception as e:
        st.warning(f"No pude generar predicción local para KPIs: {e}")
        return
    if df_pred_local is None or df_pred_local.empty:
        st.warning("Predicción local vacía; no se pueden calcular KPIs.")
        return

    # 2) Cargar mínimos y buffer
    try:
        df_param = pd.read_excel("Capacidades tanques.xlsx", sheet_name="parametros_tanques")
        minimos_por_tanque = dict(zip(df_param["Tanque"].astype(str), df_param["Mínimo permitido"].astype(float)))
    except Exception as e:
        st.error(f"No pude leer mínimos por tanque: {e}")
        return
    buffer_tanque = float(st.session_state.get("buffer_tanque_pas2", 0))

    # 3) Inventario actual
    if not os.path.exists("inventario_actual.csv"):
        st.info("Falta inventario_actual.csv. Ve a '📝 Ingreso diario...' para registrar medidas.")
        return
    df_inv_actual = pd.read_csv("inventario_actual.csv", parse_dates=["Fecha"])

    # 4) Stock útil por producto + Cobertura exacta (con PREDICCIÓN LOCAL)
    su_por_prod = stock_util_por_producto(df_inv_actual, minimos_por_tanque, buffer_tanque)

    # incluir_hoy=True ⇒ el “N días” YA cuenta el día de hoy
    df_cov, fechas_pedido = cobertura_exacta_por_producto(
        df_pred_local, su_por_prod, incluir_hoy=True
    )

    # Construir cov_info correctamente (¡todo este bloque va DENTRO del for!)
    cov_info = {}
    for _, r in df_cov.iterrows():
        p = str(r["Producto"]).strip()
        dias_txt = str(r["Cobertura_dias"]).strip()

        # Parsear la fecha de agotamiento si existe
        agot = None
        if pd.notnull(r["Fecha_agotamiento"]):
            txt_agot = str(r["Fecha_agotamiento"]).strip()
            if "✔️" not in txt_agot:  # evitar el texto "✔️ Cubierto en el horizonte"
                try:
                    agot = pd.to_datetime(txt_agot).normalize()
                except Exception:
                    agot = None

        # fin_completo = (agotamiento - 1 día), si hay agotamiento
        fin = (agot - pd.Timedelta(days=1)).normalize() if agot is not None else None

        rango = str(r["Rango_cubierto_completo"]) if "Rango_cubierto_completo" in r else ""
        cov_info[p] = {"dias_txt": dias_txt, "agot": agot, "fin": fin, "rango": rango}

    # (Opcional) Guardar cosas útiles (NO guardamos la predicción)
    st.session_state["fechas_pedido_sugeridas"] = fechas_pedido
    st.session_state["buffer_tanque_pas2"] = buffer_tanque

    # 5) KPIs por producto (independientes del slicer)
    c1, c2, c3 = st.columns(3)
    for i, prod in enumerate(["ACPM", "CORRIENTE", "SUPREME"]):
        su = float(su_por_prod.get(prod, 0.0))
        col = [c1, c2, c3][i]

        with col:
            # --- Stock útil ---
            kpi_chip(f"{prod} — Stock útil", f"{su:,.0f} gal",
                     f"Colchón por tanque: {buffer_tanque:.0f} gal")

            # --- Cobertura (incluye hoy) ---
            cov_txt = cov_info.get(prod, {}).get("dias_txt", "—")
            if cov_txt not in (None, "—"):
                cov_txt = f"{cov_txt} (incluye hoy)"
            kpi_chip(f"{prod} — Cobertura", cov_txt)

            # --- Fecha sugerida (lead time = 1 día) ---
            # Usamos fin_completo (agot - 1 día) y NUNCA sugerimos en pasado
            _hoy = pd.to_datetime("today").normalize()
            fin = cov_info.get(prod, {}).get("fin", None)
            agot = cov_info.get(prod, {}).get("agot", None)

            if fin is not None:
                fecha_pedido = max(_hoy, fin)
                sugerencia_txt = fecha_pedido.strftime("%Y-%m-%d")
            elif agot is not None:
                # fallback: por si alguna fila no trae fin
                fecha_pedido = max(_hoy, (agot - pd.Timedelta(days=1)).normalize())
                sugerencia_txt = fecha_pedido.strftime("%Y-%m-%d")
            else:
                sugerencia_txt = "Sin urgencia"

            kpi_chip(f"{prod} — Fecha sugerida (lead time = 1 día)", sugerencia_txt)






    # ============ 🚚 Pedido recomendado (arriba, 100% con predicción LOCAL) ============
    hdr_pedido = st.empty()  # placeholder del encabezado dinámico

    # 1) Horizonte objetivo automático: primer producto que se agota (desde la TABLA 'cov_info')
    hoy_ts = pd.to_datetime("today").normalize()
    hoy_d = hoy_ts.date()
    agot_list = [v["agot"] for v in cov_info.values() if v.get("agot") is not None]

    # Si hay agotamientos, usa el mínimo (más cercano); si no, conservador 2 días
    if agot_list:
        dias_obj = max(1, min((ag.date() - hoy_d).days for ag in agot_list))
    else:
        dias_obj = 2  # si nadie se agota en el horizonte, usa 2 días conservador

    # 2) Déficits por tanque y requerimiento por producto usando la PREDICCIÓN LOCAL
    req_por_prod, df_def = deficits_hasta_objetivo(
        df_pred_tanques=df_pred_local,
        df_inv_actual=df_inv_actual,
        minimos_por_tanque=minimos_por_tanque,
        buffer_tanque=buffer_tanque,
        dias_objetivo=dias_obj,
        lead_time=1,
        reserva_operativa_gal=0,
    )

    # 3) Plan para los carrotanques
    carrotanques = {"751": [1440, 1320, 880], "030": [1500, 1215, 740]}

    def uso_carro(df_plan, caps):
        if df_plan is None or df_plan.empty:
            return 0.0
        return float(df_plan["Galones asignados"].sum()) / float(sum(caps)) * 100.0

    planes = {}
    for placa, caps in carrotanques.items():
        df_plan = plan_carrotanque_3_comp(
            req_por_prod=req_por_prod,
            df_deficits=df_def,
            carrotanques_caps={placa: caps}
        )
        planes[placa] = (df_plan, caps)

    # 4) Mostrar resumen compacto
    st.caption(f"Horizonte objetivo automático: **{dias_obj} día(s)** (primer agotamiento detectado).")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("📦 Requerimiento por producto")
        st.write({k: round(v, 1) for k, v in (req_por_prod or {}).items()})
    with colB:
        st.subheader("🧯 Tanques más urgentes")
        st.dataframe(df_def, use_container_width=True, height=220)

    st.subheader("🗺️ Plan propuesto")
    mejor = None
    mejor_pct = -1.0
    for placa, (df_plan, caps) in planes.items():
        if df_plan is None or df_plan.empty:
            st.info(f"{placa}: no se pudo armar plan.")
            continue
        pct = uso_carro(df_plan, caps)
        if pct > mejor_pct:
            mejor, mejor_pct = placa, pct
        st.markdown(f"**Carrotanque {placa}** — Aprovechamiento: {pct:.1f}% (capacidad {sum(caps)} gal)")
        st.dataframe(df_plan, use_container_width=True, height=160)

    if mejor:
        st.success(f"✅ Sugerencia: usar **{mejor}** (mayor aprovechamiento).")

    # === Encabezado de pedido recomendado (lead time = 1 día) — usando la MISMA fuente (cov_info) ===
    lead_time_dias = 1
    fecha_arribo = min(agot_list) if agot_list else None

    if fecha_arribo is not None:
        fecha_pedido = (fecha_arribo - pd.Timedelta(days=lead_time_dias)).normalize()
        # Nunca sugerir en pasado
        if fecha_pedido < hoy_ts:
            fecha_pedido = hoy_ts
            fecha_arribo = (hoy_ts + pd.Timedelta(days=lead_time_dias)).normalize()

        hdr_pedido.markdown(
            f"### 🚚 Pedido recomendado — **Pide el {fecha_pedido.strftime('%Y-%m-%d')}** "
            f"(llegada {fecha_arribo.strftime('%Y-%m-%d')})"
        )

        # Guarda en sesión si otras partes lo necesitan
        st.session_state["pedido_fecha_pedir"] = fecha_pedido.date()
        st.session_state["pedido_fecha_llegar"] = fecha_arribo.date()
        st.session_state["pedido_lead_time"] = lead_time_dias
    else:
        hdr_pedido.markdown("### 🚚 Pedido recomendado — *(no es necesario pedir aún)*")

    # 5) Guardar en session_state para la explicación breve
    st.session_state["pedido_dias_objetivo"] = dias_obj
    st.session_state["df_def_pedido"] = df_def
    st.session_state["planes_pedido"] = {pl: df for pl, (df, cs) in planes.items()}
    st.session_state["caps_pedido"] = {pl: cs for pl, (df, cs) in planes.items()}
    with st.expander("📋 Explicación breve del pedido"):
        hoy_txt = hoy_ts.date()
        desde = (hoy_ts + pd.Timedelta(days=1)).date()
        st.write(f"Hoy **{hoy_txt}** · lead time **1 día** → cubrimos **{dias_obj}** día(s) desde **{desde}**.")
        if df_def is not None and not df_def.empty:
            df_exp = df_def.sort_values(["Producto", "Deficit", "StockUtilProy"], ascending=[True, False, True])
            for _, r in df_exp.iterrows():
                tanque, prod = str(r["Tanque"]), str(r["Producto"])
                su0, cons = float(r["StockUtilInicial"]), float(r["ConsumoAcum"])
                su_proj, deficit = float(r["StockUtilProy"]), float(r["Deficit"])
                if deficit > 0:
                    st.write(f"🔴 {tanque} ({prod}): SU {su0:,.0f} → {su_proj:,.0f} (consumo {cons:,.0f}) · **déficit {deficit:,.0f}**.")
                else:
                    st.write(f"🟢 {tanque} ({prod}): SU {su0:,.0f} → {su_proj:,.0f} (sin déficit).")

    st.divider()






  

resumen_estado_actual_ui(pred_dias_default=14)


# ===================== 📈 Bienvenida — Precisión del modelo =====================



reales_base = (
    df_tanques.groupby(["Fecha","Producto"])["Galones"]
    .sum().reset_index().rename(columns={"Galones":"Real"})
)

if reales_base.empty:
    st.info("No hay datos históricos por tanques para evaluar precisión.")
else:
    
    fin_bt = pd.to_datetime(reales_base["Fecha"]).max().normalize()
    ini_bt = fin_bt - pd.Timedelta(days=13)  # incluye fin_bt
    reales_14 = reales_base[(reales_base["Fecha"] >= ini_bt) & (reales_base["Fecha"] <= fin_bt)].copy()

    
    try:
        dias_bt = (fin_bt - ini_bt).days + 1
        pred_bt = generar_pred_por_tanques(df_tanques, modelo_tanques, ini_bt, dias_bt)
    except Exception as e:
        st.error(f"No fue posible generar la predicción automática de backtest (tanques): {e}")
        pred_bt = pd.DataFrame()

    if pred_bt.empty:
        st.info("No se generaron predicciones para el rango de backtest.")
    else:
        
        pred_prod = (
            pred_bt.groupby(["Fecha","Producto"])["Predicción (galones)"]
                  .sum().reset_index().rename(columns={"Predicción (galones)":"Pred"})
        )

        
        eval_df = reales_14.merge(pred_prod, on=["Fecha","Producto"], how="inner").sort_values(["Producto","Fecha"])

        if eval_df.empty:
            st.info("No hay intersección entre reales y predicción en el backtest.")
        else:
            
            eval_df["Residuo"]  = eval_df["Pred"] - eval_df["Real"]
            eval_df["AbsError"] = (eval_df["Pred"] - eval_df["Real"]).abs()
            eval_df["RelError"] = eval_df.apply(
                lambda r: (abs(r["Residuo"]) / r["Real"]) if r["Real"] not in (0, 0.0) else pd.NA, axis=1
            )

            # 🎯 MRE% por producto y general
            mre_prod = (
                eval_df.groupby("Producto")["RelError"]
                       .mean().mul(100).rename("MRE_%").reset_index()
            )
            
            eval_pos = eval_df[eval_df["Real"] > 0].copy()
            mape_total = float(eval_pos["AbsError"].div(eval_pos["Real"]).mean() * 100) if not eval_pos.empty else None

            st.markdown("### 📈 Precisión del modelo — últimos 14 días")
            st.caption(f"Comparando **{ini_bt.date()} → {fin_bt.date()}** | Fuente: modelo de **TANQUES** (auto), reales por **tanques**.")

            c1, c2, c3, c4 = st.columns(4)
            for i, prod in enumerate(["ACPM","CORRIENTE","SUPREME"]):
                val = mre_prod.loc[mre_prod["Producto"]==prod, "MRE_%"]
                val = float(val.iloc[0]) if not val.empty and pd.notna(val.iloc[0]) else None
                col = [c1,c2,c3][i]
                with col:
                    st.metric(f"{prod} — MRE%", f"{val:.1f}%" if val is not None else "—")
            with c4:
                st.metric("MAPE general", f"{mape_total:.1f}%" if mape_total is not None else "—")

            # 7) Barras de residuo por día y producto
            st.markdown("#### 📊 Residuos diarios (Pred − Real) — por producto")
            import altair as alt
            bars = alt.Chart(eval_df).mark_bar().encode(
                x=alt.X("Fecha:T", title="Fecha"),
                y=alt.Y("Residuo:Q", title="Residuo (gal)"),
                color=alt.condition("datum.Residuo > 0", alt.value("#4caf50"), alt.value("#e53935")),
                tooltip=["Producto","Fecha:T","Real:Q","Pred:Q","Residuo:Q","AbsError:Q"]
            ).properties(height=180)
            st.altair_chart(bars.facet(column=alt.Column("Producto:N", title=None)), use_container_width=True)

            # 8) Dispersión Pred vs Real con línea ideal
            st.markdown("#### 🔎 Dispersión Pred vs Real — por producto")
            lim_max = float(max(eval_df["Real"].max(), eval_df["Pred"].max()))
            line_df = pd.DataFrame({"x":[0, lim_max], "y":[0, lim_max]})
            ideal = alt.Chart(line_df).mark_line().encode(x="x:Q", y="y:Q")
            pts = alt.Chart(eval_df).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("Real:Q", title="Real (gal)"),
                y=alt.Y("Pred:Q", title="Pred (gal)"),
                color=alt.Color("Producto:N"),
                tooltip=["Producto","Fecha:T","Real:Q","Pred:Q","Residuo:Q","AbsError:Q"]
            )
            st.altair_chart(ideal + pts, use_container_width=True)










# ===================== UI: DATOS REALES =====================

with st.expander("📁 Ver datos reales cargados (turnos y tanques)"):
    st.sidebar.header("🔍 Filtros de visualización")
    fecha_default_inicio = pd.to_datetime("2025-08-01")
    fecha_default_fin    = pd.to_datetime("2025-08-31")
    fecha_inicio = st.sidebar.date_input("Fecha desde", fecha_default_inicio)
    fecha_fin    = st.sidebar.date_input("Fecha hasta", fecha_default_fin)

    productos = st.sidebar.multiselect(
        "Filtrar por producto",
        options=df_turnos["Producto"].unique(),
        default=list(df_turnos["Producto"].unique())
    )

    st.subheader("📊 Datos Reales - Consumo por Turnos")
    df_turnos_filtrado = df_turnos[
        (df_turnos["Fecha"] >= pd.to_datetime(fecha_inicio)) &
        (df_turnos["Fecha"] <= pd.to_datetime(fecha_fin)) &
        (df_turnos["Producto"].isin(productos))
    ]
    st.dataframe(df_turnos_filtrado.sort_values(["Fecha", "Turno"]), use_container_width=True)

    st.subheader("📊 Datos Reales - Consumo por Tanques")
    df_tanques_filtrado = df_tanques[
        (df_tanques["Fecha"] >= pd.to_datetime(fecha_inicio)) &
        (df_tanques["Fecha"] <= pd.to_datetime(fecha_fin)) &
        (df_tanques["Producto"].isin(productos))
    ]
    st.dataframe(df_tanques_filtrado.sort_values(["Fecha", "Tanque"]), use_container_width=True)

# ===================== UI: PREDICCIÓN POR TURNOS =====================

with st.expander("🔮 Predicción por Turnos"):
    st.markdown("**Selecciona los parámetros de predicción:**")
    fecha_inicio_turnos = st.date_input(
        "Fecha de inicio de predicción", 
        value=pd.to_datetime("2025-07-26"),
        key="fecha_turnos"
    )
    dias_a_predecir = st.slider(
        "¿Cuántos días deseas predecir?", 1, 31, 4, key="dias_turnos"
    )

    if st.button("Predecir consumo por turnos"):
        productos_list = ["ACPM", "CORRIENTE", "SUPREME"]
        preds = []
        for producto in productos_list:
            df_prod = df_turnos[df_turnos["Producto"] == producto].copy().sort_values("Fecha")
            ult = df_prod.copy()
            for i in range(dias_a_predecir):
                f = fecha_inicio_turnos + timedelta(days=i)
                dow = f.weekday()
                for turno in ["Diurno", "Nocturno"]:
                    hist = ult[ult["Turno"] == turno]
                    lag1 = hist["Galones"].iloc[-1]
                    m3   = hist["Galones"].tail(3).mean()
                    m7   = hist["Galones"].tail(7).mean()
                    pf = {
                        "Producto_ACPM":     int(producto == "ACPM"),
                        "Producto_CORRIENTE": int(producto == "CORRIENTE"),
                        "Producto_SUPREME":   int(producto == "SUPREME")
                    }
                    tf = {
                        "Turno_Diurno":     int(turno == "Diurno"),
                        "Turno_Nocturno":   int(turno == "Nocturno")
                    }
                    entrada = pd.DataFrame([{
                        "Año": f.year, "Mes": f.month, "Día": f.day,
                        "NumeroDiaSemana": dow,
                        "Lag_1": lag1,
                        "MediaMovil_3": m3, "MediaMovil_7": m7,
                        **pf, **tf
                    }])
                    y = modelo_turnos.predict(entrada)[0]
                    preds.append({
                        "Fecha": f,
                        "Turno": turno,
                        "Producto": producto,
                        "Predicción (galones)": round(y, 2)
                    })
                    ult = pd.concat([
                        ult,
                        pd.DataFrame([{
                            "Fecha": f, "Turno": turno,
                            "Producto": producto, "Galones": y
                        }])
                    ], ignore_index=True)

        df_res = pd.DataFrame(preds)
        st.success("✅ Predicción generada con éxito")
        st.dataframe(df_res, use_container_width=True)
        totales_turnos = (
            df_res.groupby("Producto")["Predicción (galones)"].sum().reset_index()
            .rename(columns={"Predicción (galones)": "Total_Galones"})
        )
        st.subheader(f"Total global predicción por turnos ({dias_a_predecir} días)")
        st.table(totales_turnos)

# ===================== UI: PREDICCIÓN POR TANQUES =====================

with st.expander("🔮 Predicción por Tanques"):
    st.markdown("**Selecciona los parámetros de predicción:**")
    inicio_tq = st.date_input("Fecha inicio predicción", key="fecha_tanq", value=pd.to_datetime("today"))
    dias_tq   = st.slider("¿Cuántos días deseas predecir?", 1, 31, 4, key="dias_tanq")

    if st.button("Predecir consumo por tanques"):
        df_pred = generar_pred_por_tanques(df_tanques, modelo_tanques, inicio_tq, dias_tq)

        if df_pred.empty:
            st.warning("No se generaron predicciones. Revisa que existan datos históricos por tanque/producto.")
            st.stop()

        st.success("✅ Predicción generada con éxito")
        st.dataframe(df_pred.sort_values(["Fecha","Producto","Tanque"]), use_container_width=True)

        
        tot_por_tanque = (
            df_pred.groupby("Tanque")["Predicción (galones)"]
            .sum().reset_index()
            .rename(columns={"Predicción (galones)": f"Total_{dias_tq}_días"})
            .sort_values(by=f"Total_{dias_tq}_días", ascending=False)
        )
        st.subheader(f"🧮 Consumo total por tanque en {dias_tq} días")
        st.dataframe(tot_por_tanque, use_container_width=True)

        
        tot_por_producto = (
            df_pred.groupby("Producto")["Predicción (galones)"]
            .sum().reset_index()
            .rename(columns={"Predicción (galones)": f"Total_{dias_tq}_días"})
            .sort_values(by=f"Total_{dias_tq}_días", ascending=False)
        )
        st.subheader(f"🧮 Consumo total por producto en {dias_tq} días")
        st.dataframe(tot_por_producto, use_container_width=True)

        
        st.session_state["df_pred_tanques"] = df_pred
        st.session_state["tot_por_tanque"] = tot_por_tanque
        st.session_state["tot_por_producto"] = tot_por_producto



# ===================== UI: INGRESO DIARIO =====================

def mapear_tanque_a_hoja(nombre_tanque: str) -> str:
    """
    Mapea el nombre del tanque a la hoja del aforo unificado.
    Ejemplos válidos en el Excel: '6 ACPM', '7 ACPM', '3 CTE', '5 CTE', 'SUPREME'
    """
    n = str(nombre_tanque).upper()
    if "ACPM" in n:
        return "6 ACPM" if "6" in n else "7 ACPM"
    if "CORRIENTE" in n or "CTE" in n:
        return "3 CTE" if "3" in n else "5 CTE"
    return "SUPREME"


def convertir_cm_a_galones(tanque: str, medida_cm: float, producto: str) -> float:
    """
    Convierte una medición (cm) a galones usando aforos_unificado.xlsx.
    - Para SUPREME: la columna ya está en centímetros → usa 'medida_cm' tal cual.
    - Para ACPM y CORRIENTE: el aforo está en milímetros → usa 'medida_cm * 10'.
    Hace interpolación lineal entre dos puntos de la tabla.
    """
    import pandas as pd

    hoja = mapear_tanque_a_hoja(tanque)
    try:
        aforos = pd.read_excel("aforos_unificado.xlsx", sheet_name=None)
    except Exception:
        return 0.0

    tabla = aforos.get(hoja)
    if tabla is None or "Milimetros" not in tabla.columns or "Galones" not in tabla.columns:
        return 0.0

    
    is_supreme = str(producto).upper().strip() in ["SUPREME", "MAX PRO", "SUPREME MAX PRO"]
    x = float(medida_cm) if is_supreme else float(medida_cm) * 10.0  # cm→mm para ACPM/CTE

    tabla = tabla.dropna(subset=["Milimetros", "Galones"]).sort_values("Milimetros")
    if tabla.empty:
        return 0.0

    
    if x <= float(tabla["Milimetros"].min()):
        return float(tabla["Galones"].iloc[0])
    if x >= float(tabla["Milimetros"].max()):
        return float(tabla["Galones"].iloc[-1])

    
    lo = tabla[tabla["Milimetros"] <= x].iloc[-1]
    hi = tabla[tabla["Milimetros"] >= x].iloc[0]
    if float(hi["Milimetros"]) == float(lo["Milimetros"]):
        return float(lo["Galones"])

    m_lo, g_lo = float(lo["Milimetros"]), float(lo["Galones"])
    m_hi, g_hi = float(hi["Milimetros"]), float(hi["Galones"])
    gal = g_lo + (x - m_lo) * (g_hi - g_lo) / (m_hi - m_lo)
    return float(round(gal, 2))



with st.expander("📝 Ingreso diario de medidas de tanques"):
    
    if not os.path.exists("inventario_actual.csv"):
        _ = cargar_inventario_actual()

    df_inv_actual = pd.read_csv("inventario_actual.csv", parse_dates=["Fecha"])
    st.subheader("📋 Inventario actual (antes de guardar):")
    st.dataframe(df_inv_actual, use_container_width=True)

    tanques = df_tanques["Tanque"].unique()
    col1, col2, col3 = st.columns(3)
    with col1:
        tanque = st.selectbox("Selecciona el tanque", tanques)
    with col2:
        fecha_med = st.date_input("Fecha de medición", value=pd.to_datetime("today"))
    with col3:
        medida_cm = st.number_input("Medida actual (cm)", min_value=0.0, step=0.1)

    
    if not df_inv_actual[df_inv_actual["Tanque"] == tanque].empty:
        producto = df_inv_actual.loc[df_inv_actual["Tanque"] == tanque, "Producto"].iloc[0]
        st.info(f"Producto detectado: **{producto}**")
    else:
        producto = detectar_producto_desde_tanque(str(tanque))
        st.info(f"(Estimado) Producto: **{producto}**")

    if st.button("Guardar medida"):
        gal = convertir_cm_a_galones(tanque, medida_cm, producto)
        df_inv = pd.read_csv("inventario_actual.csv", parse_dates=["Fecha"])
        mask = (df_inv["Tanque"] == tanque) & (df_inv["Producto"] == producto)
        if mask.any():
            df_inv.loc[mask, ["Medida_cm", "Galones", "Fecha"]] = [medida_cm, gal, pd.to_datetime(fecha_med)]
        else:
            nueva = {"Tanque": tanque, "Producto": producto, "Medida_cm": medida_cm, "Galones": gal, "Fecha": pd.to_datetime(fecha_med)}
            df_inv = pd.concat([df_inv, pd.DataFrame([nueva])], ignore_index=True)
        df_inv.to_csv("inventario_actual.csv", index=False)
        st.success(f"✅ Inventario actualizado: {gal} galones en {tanque} ({producto}) al {fecha_med}")
        st.subheader("📋 Inventario actual (después de guardar):")
        st.dataframe(df_inv[["Fecha", "Tanque", "Producto", "Medida_cm", "Galones"]], use_container_width=True)


with st.expander("🧯 Cobertura exacta y fecha de pedido (lead time = 1 día)"):
    
    try:
        inicio_cov = pd.to_datetime("today").normalize()
        df_pred_cov = generar_pred_por_tanques(df_tanques, modelo_tanques, inicio_cov, 14)  # puedes ajustar 14 si quieres
    except Exception as e:
        st.warning(f"No fue posible preparar predicción local para Cobertura exacta: {e}")
        df_pred_cov = pd.DataFrame()

    if df_pred_cov is None or df_pred_cov.empty:
        st.warning("No hay predicciones locales por tanques para calcular cobertura exacta.")
    else:
        # 2) Parámetros: buffer y mínimos (lead time = 1 fijo aquí)
        buffer_tanque = st.number_input(
            "Colchón por tanque (gal)",
            min_value=0.0, value=float(st.session_state.get("buffer_tanque_pas2", 0)),
            step=50.0
        )

        # Cargar mínimos de Excel
        try:
            df_param = pd.read_excel("Capacidades tanques.xlsx", sheet_name="parametros_tanques")
            minimos_por_tanque = dict(zip(
                df_param["Tanque"].astype(str),
                df_param["Mínimo permitido"].astype(float)
            ))
        except Exception as e:
            st.error(f"No pude leer mínimos por tanque: {e}")
            minimos_por_tanque = {}

        # 3) Inventario actual
        if not os.path.exists("inventario_actual.csv"):
            st.info("No existe inventario_actual.csv. Ve a 'Ingreso diario de medidas' y guarda al menos una medición.")
        else:
            df_inv_actual = pd.read_csv("inventario_actual.csv", parse_dates=["Fecha"])

            # 4) Stock útil por PRODUCTO (bandeo)
            su_por_prod = stock_util_por_producto(df_inv_actual, minimos_por_tanque, buffer_tanque)

            # 5) Cobertura exacta con PREDICCIÓN LOCAL
            df_cov, fechas_pedido = cobertura_exacta_por_producto(
                df_pred_cov, su_por_prod, incluir_hoy=True
            )

            st.subheader("⛽ Cobertura exacta por producto (incluye hoy)")
            st.dataframe(df_cov, use_container_width=True)

            # Guardamos solo resultados (no la predicción)
            st.session_state["df_cov_exacta"] = df_cov
            st.session_state["fechas_pedido_sugeridas"] = fechas_pedido
            st.session_state["buffer_tanque_pas2"] = buffer_tanque
            






                








