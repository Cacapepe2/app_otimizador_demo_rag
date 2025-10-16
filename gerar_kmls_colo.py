import pandas as pd
import numpy as np
import os
from geopy.distance import distance as geopy_distance
from geopy.point import Point
import simplekml

def gerar_circulo_kml(latitude, longitude, raio_metros, nome_arquivo, nome_ponto, proprietario, id_detentora):
    """
    Gera um arquivo KML com um círculo de raio especificado ao redor de um ponto.
    
    Args:
        latitude: Latitude do centro (torre de colo)
        longitude: Longitude do centro (torre de colo)
        raio_metros: Raio do círculo em metros
        nome_arquivo: Nome do arquivo KML a ser salvo
        nome_ponto: Nome/descrição do ponto (UFSIGLA)
        proprietario: Nome do proprietário da torre
        id_detentora: ID da detentora
    """
    # Cria objeto KML
    kml = simplekml.Kml()
    
    # Centro do círculo
    centro = Point(latitude, longitude)
    
    # Gera 36 pontos ao redor (a cada 10 graus)
    pontos_circulo = []
    for angulo in range(0, 360, 10):
        # Calcula ponto a X metros de distância no ângulo especificado
        ponto = geopy_distance(meters=raio_metros).destination(centro, angulo)
        pontos_circulo.append((ponto.longitude, ponto.latitude))
    
    # Fecha o círculo voltando ao primeiro ponto
    pontos_circulo.append(pontos_circulo[0])
    
    # Cria polígono no KML
    pol = kml.newpolygon(name=nome_ponto)
    pol.outerboundaryis = pontos_circulo
    
    # Estilização
    pol.style.linestyle.color = simplekml.Color.green
    pol.style.linestyle.width = 3
    pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.green)
    
    # Adiciona descrição detalhada
    pol.description = f"""
    <b>Torre de Colo</b><br/>
    Site: {nome_ponto}<br/>
    Proprietário: {proprietario}<br/>
    ID Detentora: {id_detentora}<br/>
    Raio: {raio_metros}m<br/>
    Centro: {latitude:.6f}, {longitude:.6f}
    """
    
    # Salva arquivo
    kml.save(nome_arquivo)
    return True

def encontrar_colo_mais_proximo(lat_site, lon_site, b_sharing, distancia_maxima_metros=500):
    """
    Encontra a torre de colo mais próxima de um site.
    
    Args:
        lat_site: Latitude do site
        lon_site: Longitude do site
        b_sharing: DataFrame com torres de colo
        distancia_maxima_metros: Distância máxima para considerar (padrão 500m)
    
    Returns:
        dict com dados da torre mais próxima ou None
    """
    if pd.isna(lat_site) or pd.isna(lon_site):
        return None
    
    # Pega todas as torres (SEM filtro de compartilhável)
    torres_disponiveis = b_sharing.copy()
    
    if torres_disponiveis.empty:
        return None
    
    # Calcula distância para cada torre
    distancias = []
    site_coord = (lat_site, lon_site)
    
    for idx, torre in torres_disponiveis.iterrows():
        if pd.notna(torre["Latitude"]) and pd.notna(torre["Longitude"]):
            torre_coord = (torre["Latitude"], torre["Longitude"])
            dist_m = geopy_distance(site_coord, torre_coord).meters
            
            if dist_m <= distancia_maxima_metros:
                distancias.append({
                    'index': idx,
                    'distancia': dist_m,
                    'latitude': torre["Latitude"],
                    'longitude': torre["Longitude"],
                    'proprietario': torre["Proprietário"],
                    'id_detentora': torre["ID Detentora"]
                })
    
    if not distancias:
        return None
    
    # Retorna a mais próxima
    mais_proxima = min(distancias, key=lambda x: x['distancia'])
    return mais_proxima

def processar_e_gerar_kmls(df_final, b_sharing, distancia_colo_metros=500):
    """
    Processa sites, encontra colos e gera KMLs + planilha para carregar_KML.py
    
    Args:
        df_final: DataFrame com sites processados
        b_sharing: DataFrame com torres de colo
        distancia_colo_metros: Distância máxima para buscar colo
    
    Returns:
        tuple: (pasta_kmls, arquivo_planilha, total_kmls_gerados)
    """
    # Cria pasta de saída
    pasta_kmls = "Carregar_poligonos"
    os.makedirs(pasta_kmls, exist_ok=True)
    
    # Lista para planilha de saída
    dados_planilha = []
    kmls_gerados = 0
    
    print("🔄 Processando sites e gerando KMLs...")
    
    for idx, row in df_final.iterrows():
        id_master = row.get("ID MASTER", "")
        sigla_sugerida = row.get("Sigla Sugerida", "")
        lat_site = row.get("Latitude")
        lon_site = row.get("Longitude")
        
        # Valida UFSIGLA
        if pd.isna(sigla_sugerida) or sigla_sugerida == "":
            uf_sigla = id_master[:5] if len(str(id_master)) >= 5 else str(id_master)
        else:
            uf_sigla = str(sigla_sugerida)
        
        # Busca colo mais próximo
        colo = encontrar_colo_mais_proximo(lat_site, lon_site, b_sharing, distancia_colo_metros)
        
        if colo:
            # Gera KML
            nome_kml = f"{uf_sigla}.kml"
            caminho_kml = os.path.join(pasta_kmls, nome_kml)
            
            try:
                sucesso = gerar_circulo_kml(
                    latitude=colo['latitude'],
                    longitude=colo['longitude'],
                    raio_metros=100,
                    nome_arquivo=caminho_kml,
                    nome_ponto=uf_sigla,
                    proprietario=colo['proprietario'],
                    id_detentora=colo['id_detentora']
                )
                
                if sucesso:
                    kmls_gerados += 1
                    processado = "Sim"
                    print(f"✅ KML gerado: {nome_kml} (Colo: {colo['proprietario']} a {colo['distancia']:.0f}m)")
                else:
                    processado = "Erro ao gerar KML"
            except Exception as e:
                processado = f"Erro: {str(e)}"
                print(f"❌ Erro ao gerar {nome_kml}: {str(e)}")
        else:
            processado = "OK"
        
        # Adiciona à planilha
        dados_planilha.append({
            "ID MASTER": id_master,
            "UFSIGLA": uf_sigla,
            "Processado pela Automação": processado
        })
    
    # Salva planilha para carregar_KML.py
    df_saida = pd.DataFrame(dados_planilha)
    arquivo_planilha = "Planilha_Carregar_KML.xlsx"
    df_saida.to_excel(arquivo_planilha, index=False)
    
    print(f"\n✅ Processamento concluído!")
    print(f"📁 Pasta: {pasta_kmls}")
    print(f"📊 Planilha: {arquivo_planilha}")
    print(f"📍 KMLs gerados: {kmls_gerados}/{len(df_final)}")
    
    return pasta_kmls, arquivo_planilha, kmls_gerados

if __name__ == "__main__":
    # Teste standalone
    try:
        df = pd.read_excel("Gestão_SOI.xlsx")
        b_sharing = pd.read_excel("BASE_SHARING_FLY_JUL 2025 v2 2.xlsx",
            usecols=["UF", "ID Detentora","Latitude", "Longitude","Altitude", "altura Disponível","Proprietário","Compartilhável"])
        processar_e_gerar_kmls(df, b_sharing)
    except Exception as e:
        print(f"Erro: {e}")
