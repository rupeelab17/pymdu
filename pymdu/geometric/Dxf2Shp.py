import math
import os
import subprocess
import sys

from osgeo import ogr, osr, gdal
import geopandas as gpd
import pandas as pd

def create_circle(center, radius, num_segments=36):
    """
    Créer un polygone approximant un cercle.

    :param center: Tuple (x, y) représentant les coordonnées du centre.
    :param radius: Rayon du cercle.
    :param num_segments: Nombre de segments pour approximer le cercle.
    :return: Géométrie OGR de type POLYGON.
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    angle_step = 2 * math.pi / num_segments

    for i in range(num_segments):
        angle = i * angle_step
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        ring.AddPoint(x, y)

    ring.CloseRings()
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def calculate_centroid(geometry):
    """
    Calculer le centroïde ou le point moyen d'une géométrie MultiLineString25D.

    :param geometry: Géométrie OGR de type wkbMultiLineString25D.
    :return: Tuple (x, y) représentant le centroïde.
    """
    if geometry.GetGeometryType() == ogr.wkbMultiLineString25D:
        num_points = 0
        x_sum = 0
        y_sum = 0

        for i in range(geometry.GetGeometryCount()):
            sub_geometry = geometry.GetGeometryRef(i)
            for j in range(sub_geometry.GetPointCount()):
                point = sub_geometry.GetPoint(j)
                x_sum += point[0]
                y_sum += point[1]
                num_points += 1

        if num_points > 0:
            return (x_sum / num_points, y_sum / num_points)
        else:
            raise ValueError('La géométrie MultiLineString25D est vide.')
    else:
        raise ValueError("La géométrie n'est pas de type wkbMultiLineString25D.")


def calculate_circumscribed_circle_radius(geometry):
    """
    Calculer le rayon du cercle circonscrit à une géométrie donnée.

    :param geometry: Géométrie OGR (par ex., wkbMultiLineString25D).
    :return: Rayon du cercle circonscrit.
    """
    if geometry is None or geometry.GetGeometryCount() == 0:
        raise ValueError('La géométrie est vide ou invalide.')

    # Récupérer tous les points de la géométrie
    points = []
    for i in range(geometry.GetGeometryCount()):
        sub_geometry = geometry.GetGeometryRef(i)
        for j in range(sub_geometry.GetPointCount()):
            point = sub_geometry.GetPoint(j)
            points.append((point[0], point[1]))  # On ignore la dimension Z

    if not points:
        raise ValueError('Aucun point trouvé dans la géométrie.')

    # Trouver les coordonnées min et max pour définir les extrêmes
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    # Calculer le centre et le rayon du cercle circonscrit
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Rayon = distance maximale du centre à un point
    radius = max(
        math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in points
    )

    return radius, (center_x, center_y)


def get_similarity(text):
    import spacy

    text = text.replace('ENON-', '').capitalize()
    try:
        # Charger un modèle de langue pré-entraîné
        nlp = spacy.load('fr_core_news_md')  # Modèle français avec vecteurs de mots
        print("Le modèle 'fr_core_news_md' est correctement chargé.")
    except OSError as e:
        # Si le modèle n'est pas trouvé
        if "Can't find model 'fr_core_news_md'" in str(e):
            print(
                "Le modèle 'fr_core_news_md' n'est pas installé. Téléchargement en cours..."
            )
            try:
                subprocess.run(
                    [sys.executable, '-m', 'spacy', 'download', 'fr_core_news_md'],
                    check=True,
                )
                print(
                    'Téléchargement terminé. Vous pouvez maintenant utiliser le modèle.'
                )
                # Recharger après téléchargement
                nlp = spacy.load('fr_core_news_md')
            except subprocess.CalledProcessError:
                print(
                    'Erreur lors du téléchargement du modèle. Veuillez vérifier votre connexion ou votre configuration.'
                )
        else:
            # Autres erreurs éventuelles
            print(f"Une erreur inattendue s'est produite : {e}")

    table = {
        'Bâtiment': 'batiment',
        'Construction': 'batiment',
        'Zone imperméable': 'sol',
        'Zone perméable': 'sol',
        'Piscine': 'eau',
        'Serre': 'sol',
        'Sol nu': 'sol',
        'Surface eau': 'eau',
        'Neige': 'sol',
        'Conifère': 'arbre',
        'Arbre': 'arbre',
        'Feuillu': 'arbre',
        'Coupe': 'vegetation',
        'Broussaille': 'vegetation',
        'Pelouse': 'vegetation',
        'Culture': 'vegetation',
        'Terre labourée': 'sol',
        'Vigne': 'vegetation',
        'Gazon': 'vegetation',
    }
    similarity_best: float = 0.0
    tag_best: str = ''
    for k, v in table.items():
        # Convertir les mots en objets spaCy
        word1 = nlp(text)
        word2 = nlp(k)
        # Calculer la similarité
        similarity = word1.similarity(word2)
        print(f"Similarité entre '{text}' et '{k}' : {similarity:.2f}")
        if similarity > 0.6:
            similarity_best = similarity
            tag_best = v

    return tag_best, similarity_best


def dxf_to_polygon_shp(input_dxf, output_shp, encoding = 'UTF-8'):
    """
    Convertir des polygones d'un fichier DXF en un fichier SHP.

    :param input_dxf: Chemin vers le fichier DXF d'entrée.
    :param output_shp: Chemin vers le fichier shapefile de sortie.
    :param encoding: Character encoding le fichier DXF
    """
    # Charger le fichier DXF
    gdal.SetConfigOption('DXF_ENCODING', encoding)
    dxf_driver = ogr.GetDriverByName('DXF')
    dxf_data = dxf_driver.Open(input_dxf, 0)  # Mode lecture

    if not dxf_data:
        raise FileNotFoundError(f"Impossible d'ouvrir le fichier DXF : {input_dxf}")

    # Ouvrir la couche DXF
    dxf_layer = dxf_data.GetLayer()
    spatial_ref = dxf_layer.GetSpatialRef()  # Obtenir la référence spatiale

    if spatial_ref is not None:
        print('Système de projection :', spatial_ref.ExportToWkt())
    else:
        print('Aucun système de projection trouvé.')

    if not dxf_layer:
        raise RuntimeError('Aucune couche trouvée dans le fichier DXF.')

    # Créer le fichier de sortie SHP
    gdal.SetConfigOption('SHAPE_ENCODING', encoding)
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_shp):
        shp_driver.DeleteDataSource(output_shp)

    shp_data = shp_driver.CreateDataSource(output_shp)

    # Définir le système de coordonnées
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3946)

    shp_polygon_layer = shp_data.CreateLayer('polygons', spatial_ref, ogr.wkbPolygon)

    # Copier les attributs
    dxf_layer_defn = dxf_layer.GetLayerDefn()
    for i in range(dxf_layer_defn.GetFieldCount()):
        field_defn = dxf_layer_defn.GetFieldDefn(i)
        shp_polygon_layer.CreateField(field_defn)

    # Filtrer et ajouter uniquement les polygones
    for feature in dxf_layer:
        geometry = feature.GetGeometryRef()
        if geometry:
            geom_type = geometry.GetGeometryType()
            print(feature.Layer)
            if 'ENON-arbres' in feature.Layer:
                print(geom_type)
                new_feature = ogr.Feature(shp_polygon_layer.GetLayerDefn())
                # Rayon du cercle
                radius, _ = calculate_circumscribed_circle_radius(geometry)
                center = calculate_centroid(geometry)
                print('center', center)
                new_geometry = create_circle(center, radius)
                new_feature.SetGeometry(new_geometry)
            else:
                new_feature = ogr.Feature(shp_polygon_layer.GetLayerDefn())
                new_feature.SetGeometry(geometry.Clone())

            # Copier les attributs
            for i in range(feature.GetFieldCount()):
                new_feature.SetField(i, feature.GetField(i))

            shp_polygon_layer.CreateFeature(new_feature)
            new_feature = None

    # Nettoyer les ressources
    dxf_data.Destroy()
    shp_data.Destroy()
    print(f'Conversion réussie : {output_shp}')

def dxf_to_polygon_shp2(input_dxf, output_shp, tree_shp):
    """
    Convertir des polygones d'un fichier DXF en deux fichiers SHP :
    - `output_shp` pour les polygones généraux.
    - `tree_shp` pour les arbres identifiés.

    :param input_dxf: Chemin vers le fichier DXF d'entrée.
    :param output_shp: Chemin vers le fichier shapefile de sortie général.
    :param tree_shp: Chemin vers le fichier shapefile des arbres.
    """
    # Charger le fichier DXF
    dxf_driver = ogr.GetDriverByName('DXF')
    dxf_data = dxf_driver.Open(input_dxf, 0)  # Mode lecture

    if not dxf_data:
        raise FileNotFoundError(f"Impossible d'ouvrir le fichier DXF : {input_dxf}")

    # Ouvrir la couche DXF
    dxf_layer = dxf_data.GetLayer()
    spatial_ref = dxf_layer.GetSpatialRef()  # Obtenir la référence spatiale

    if spatial_ref is not None:
        print('Système de projection :', spatial_ref.ExportToWkt())
    else:
        print('Aucun système de projection trouvé.')

    if not dxf_layer:
        raise RuntimeError('Aucune couche trouvée dans le fichier DXF.')

    # Créer les fichiers de sortie SHP
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')

    # Supprimer les fichiers existants
    for shp in [output_shp, tree_shp]:
        if os.path.exists(shp):
            shp_driver.DeleteDataSource(shp)

    # Créer les sources de données de sortie
    output_data = shp_driver.CreateDataSource(output_shp)
    tree_data = shp_driver.CreateDataSource(tree_shp)

    # Définir le système de coordonnées
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3946)

    # Créer les couches
    output_layer = output_data.CreateLayer('polygons', spatial_ref, ogr.wkbPolygon)
    tree_layer = tree_data.CreateLayer('trees', spatial_ref, ogr.wkbPolygon)

    # Copier les attributs
    dxf_layer_defn = dxf_layer.GetLayerDefn()
    for i in range(dxf_layer_defn.GetFieldCount()):
        field_defn = dxf_layer_defn.GetFieldDefn(i)
        output_layer.CreateField(field_defn)
        tree_layer.CreateField(field_defn)

    # Filtrer et ajouter les polygones dans les couches appropriées
    for feature in dxf_layer:
        geometry = feature.GetGeometryRef()
        if geometry:
            # Vérifier si la couche concerne les arbres
            if 'ENON-arbres' in feature.GetField('Layer'):  # Vérifier le nom de la couche
                new_feature = ogr.Feature(tree_layer.GetLayerDefn())
                new_feature.SetGeometry(geometry.Clone())

                # Copier les attributs
                for i in range(feature.GetFieldCount()):
                    new_feature.SetField(i, feature.GetField(i))

                tree_layer.CreateFeature(new_feature)
                new_feature = None
            else:
                # Ajouter les autres polygones au shapefile général
                new_feature = ogr.Feature(output_layer.GetLayerDefn())
                new_feature.SetGeometry(geometry.Clone())

                # Copier les attributs
                for i in range(feature.GetFieldCount()):
                    new_feature.SetField(i, feature.GetField(i))

                output_layer.CreateFeature(new_feature)
                new_feature = None

    # Nettoyer les ressources
    dxf_data.Destroy()
    output_data.Destroy()
    tree_data.Destroy()

    print(f'Conversion réussie : {output_shp} et {tree_shp}')

from pymdu.geometric.Cosia import Cosia
def dxf_to_cosia_and_weighted_layers (input_shp_dxf, output_shp_gdf, bbox_coords = None, bbox_crs = 'EPSG:4326', encoding = 'UTF-8'):
    """
        Convertir des layers d'un fichier DXF en layers Cosia avec poids :
        :param input_shp_dxf: Chemin vers le fichier SHP d'entrée.
        :param output_shp: Chemin vers le fichier shapefile de sortie
        :param bbox_coords: choisir le taille
        :param bbox_crs: le crs du bbox_coords
        :param encoding: encoding de le fichier SHP
    """
    layer_properties = {'ENON-COUPES': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-arbres 12-14': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres 25-30': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres TRB 12-14': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres existants': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres multitroncs': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres multitroncs 200-250': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres tige 12-14': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres tige 25-30': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-arbres tige fruitiers 12-14': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-boules de granite': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-clôtures bois opaques': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-copeaux bois Jeux': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-cotations': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-couvre-sol massif': {"classe": "Zone imperméable", "weight": 1.0},
                        "ENON-couvre-sol pied d'arbre": {"classe": "Pelouse", "weight": 50.0},
                        'ENON-engazonnement': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-fonte': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-jeux': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-massif arbustif bas': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-massifs arbustifs hauts': {"classe": "Feuillu", "weight": 100.0},
                        'ENON-massifs boisés micro-forêts': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-minéral': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-mobilier bois': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-pavés joints enherbés': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-plantations existantes': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-plantations existantes à compléter': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-ponton estrade terrasse bois': {"classe": "Zone imperméable", "weight": 1.0},
                        'ENON-potager (terre végétale)': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-sol vert': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-vert': {"classe": "Pelouse", "weight": 50.0},
                        'ENON-école sols souple jeux': {"classe": "Zone imperméable", "weight": 1.0},
                        'MUR': {"classe": "Zone imperméable", "weight": 1.0}}
    try:
        dxf_gdf = gpd.read_file(input_shp_dxf, encoding = encoding)
        if dxf_gdf.empty:
            print(f"Le fichier SHP de DXF : {input_shp_dxf} est vide")
        else:
            print(f"Le fichier SHP de DXF : {input_shp_dxf} est ok")
    except Exception as e:
        print(f"Error reading file: {e}")

    dxf_gdf = dxf_gdf.set_crs('epsg:3946')

    if bbox_coords is not None:
        dxf_gdf = dxf_gdf.to_crs(bbox_crs)
        dxf_gdf = dxf_gdf.clip(bbox_coords)
        # dxf_gdf.to_file('../../demos/demo_paysage_gdf.shp', driver = 'ESRI Shapefile')

    dxf_gdf["classe"] = dxf_gdf["Layer"].map(lambda x: layer_properties[x]["classe"])
    dxf_gdf['color'] = [Cosia().table_color_cosia[x] for x in dxf_gdf["classe"]]
    dxf_gdf["weight"] = dxf_gdf["Layer"].map(lambda x: layer_properties[x]["weight"])
    layer_groups = {layer: data for layer, data in dxf_gdf.groupby('Layer')}

    final_gdf = gpd.GeoDataFrame(columns=dxf_gdf.columns)

    for layer, properties in sorted(layer_properties.items(), key=lambda x: x[1]["weight"], reverse=True):
        try:
            layer_data = layer_groups[layer]  # Access the corresponding layer
            layer_data['geometry'] = layer_data['geometry'].buffer(0)
            if not final_gdf.empty:
                # Remove the intersection with existing final_gdf data
                layer_data = gpd.overlay(layer_data, final_gdf, how='difference')

            # Use pandas.concat() instead of append
            final_gdf = pd.concat([final_gdf, layer_data], ignore_index=True)
        except KeyError:
            print(f"Layer '{layer}' ne se trouve pas dans layer_groups.")

    output_dir = os.path.dirname(output_shp_gdf)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)

    final_gdf.to_file(output_shp_gdf, driver='ESRI Shapefile', encoding = encoding)
    print(f"Output shapefile {output_shp_gdf} a ete cree.")
    return final_gdf

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # import geopandas as gpd

    # Afficher les géométries avec Matplotlib
    fig, ax = plt.subplots()
    input_dxf_path = '../../demos/20240911-GSPuyMou-CC46-PAYSAGE.dxf'
    output_shp_path = '../../demos/example.shp'
    tree_shp  = '../../demos/trees.shp'
    output_shp_gdf = '../../demos/example_with_layers.shp'

    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf_cut = gpd.read_file('../../demos/example_cut.shp')
    gdf_cut = gdf_cut.set_crs('epsg:3946')
    bbox_coords = gdf_cut.total_bounds

    dxf_to_polygon_shp(input_dxf_path, output_shp_path)
    dxf_gdf = dxf_to_cosia_and_weighted_layers(output_shp_path, output_shp_gdf, bbox_coords, gdf_cut.crs)
    dxf_gdf.plot(color = dxf_gdf['color'])
    plt.show()
    print("test")
    # dxf_to_polygon_shp2(input_dxf_path, output_shp_path, tree_shp)
    # gdf = gpd.read_file(output_shp_path, driver="ESRI Shapefile")
    # gdf.plot()
    # plt.show()

    # print(get_similarity('ENON-arbres multitroncs 200-250'))
    # print(get_similarity('ENON-arbres tige fruitiers 12-14'))
    # print(get_similarity('ENON-arbres TRB 12-14'))
    # print(get_similarity('engazonnement'))
    # print(get_similarity('ENON-construction'))
