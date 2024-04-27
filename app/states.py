import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import Role
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier

from utils import read_config, write_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config()


@dataclass
class Patient:
    patient_id: str
    diseases: [str]
    genes: [str]
    proteins: [str]
    phenotypes: [str]
    is_sick: bool

    def __init__(self, pa, di, ge, pr, ph):
        self.patient_id = pa
        self.diseases = di
        self.genes = ge
        self.proteins = pr
        self.phenotypes = ph
        self.is_sick = self.diseases != ['control'] and len(self.diseases)


@dataclass
class TrainRow:
    is_sick: bool
    features: np.array

    def __init__(self, health, fts: [int]):
        self.is_sick = health
        self.features = np.array(fts)


@dataclass
class ValRow:
    patient_id: str
    features: np.array

    def __init__(self, patient_id, fts: [int]):
        self.patient_id = patient_id
        self.features = np.array(fts)


@app_state('initial')
class ExecuteState(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self):
        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"\nNeo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        logger.info(f'The neo4j_credentials are: {neo4j_credentials}')
        if True:
            # Driver instantiation
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

            # Create a driver session with defined DB
            with driver.session(database=NEO4J_DB) as session:

                query = (
                    "MATCH (b:Biological_sample)-[:HAS_DISEASE]->() "
                    "RETURN DISTINCT b.subjectid"
                )
                result = session.run(query)
                relevant_subjects = np.array([record["b.subjectid"] for record in result])
                relevant_subjects = np.unique(relevant_subjects)

                query_val = (
                    "MATCH (b:Biological_sample) "
                    "WHERE NOT (b)-[:HAS_DISEASE]->() "
                    "RETURN b.subjectid"
                )
                result = session.run(query_val)
                val_subjects = np.array([record["b.subjectid"] for record in result])
                val_subjects = np.unique(val_subjects)

                data_total = []
                patients_props = []
                patients_health = []
                # patients, rows = list(), list()
                logger.info(f'run {len(relevant_subjects)} queries')
                for patient_id in relevant_subjects:
                    query_info = (
                        "MATCH (b:Biological_sample {subjectid: $patient_id}) "
                        "OPTIONAL MATCH (b)-[:HAS_DISEASE]->(d:Disease) "
                        "OPTIONAL MATCH (b)-[:HAS_PROTEIN]->(p:Protein) "
                        "OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype) "
                        "OPTIONAL MATCH (b)-[:HAS_DAMAGE]->(g:Gene) "
                        "RETURN b, "
                        "COLLECT(DISTINCT CASE WHEN EXISTS(d.id) THEN d.id ELSE d.name END) AS diseases_synonyms, "
                        "COLLECT(DISTINCT g.id) AS genes_ids, "
                        "COLLECT(DISTINCT p.id) AS proteins_ids, "
                        "COLLECT(DISTINCT ph.id) AS phenotypes_ids"
                    )
                    result_info = session.run(query_info, patient_id=patient_id)
                    result_info = result_info.single()
                    # diseases, genes, proteins, phenotypes = result_info[1:5]
                    # healthy = diseases == ['control']
                    # rows.append(Row())
                    # patients.append(Patient(patient, *result_info[1:5]))

                    patient_props = result_info[2] + result_info[3] + result_info[4]
                    patient_health = int(not result_info[1] == ['control'])
                    data_total.append([patient_props, patient_health])
                    patients_props.extend(patient_props)
                    patients_health.append(patient_health)
                logger.info('train set queries done')

                data_val = []
                for patient_id in val_subjects:
                    query_info = (
                        "MATCH (b:Biological_sample {subjectid: $patient_id}) "
                        "OPTIONAL MATCH (b)-[:HAS_PROTEIN]->(p:Protein) "
                        "OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype) "
                        "OPTIONAL MATCH (b)-[:HAS_DAMAGE]->(g:Gene) "
                        "RETURN b, "
                        "COLLECT(DISTINCT g.id) AS genes_ids, "
                        "COLLECT(DISTINCT p.id) AS proteins_ids, "
                        "COLLECT(DISTINCT ph.id) AS phenotypes_ids"
                    )
                    result_info = session.run(query_info, patient_id=patient_id)
                    result_info = result_info.single()
                    patient_props = result_info[1] + result_info[2] + result_info[3]
                    data_val.append([patient_props, patient_id])
                    patients_props.extend(patient_props)
                    # patients_health.extend(result_info[1])
                logger.info('val set queries done')

            patients_props = np.unique(np.array(patients_props))
            patients_health = np.array(patients_health)
            property_dict = {value: index for index, value in enumerate(patients_props)}
            # disease_dict = {value: index for index, value in enumerate(patients_health)}
            data_binary = []

            # encode the ragged string lists as a binary numpy array
            for patient_props, patient_health in data_total:
                p_list = [property_dict[value] for value in patient_props]
                binary_p = [0] * len(patients_props)
                for index in p_list:
                    binary_p[index] = 1

                # d_list = [disease_dict[value] for value in patient_health]
                # binary_d = [0] * len(disease_dict)
                # for index in d_list:
                #     binary_d[index] = 1
                data_binary.append([patient_health, binary_p])
            logger.info(f'data_binary has length {len(data_binary)}')

            val_binary = []
            for patient_props, patient_id in data_val:
                p_list = [property_dict[value] for value in patient_props]
                binary_p = [0] * len(patients_props)
                for index in p_list:
                    binary_p[index] = 1
                val_binary.append([patient_id, binary_p])
            logger.info('encoding done')

            with open("data_binary", "wb") as fp:
                pickle.dump(data_binary, fp)
            with open("val_binary", "wb") as fp:
                pickle.dump(val_binary, fp)
            with open("property_dict", "wb") as fp:
                pickle.dump(property_dict, fp)
            # with open("disease_dict", "wb") as fp:
            #     pickle.dump(disease_dict, fp)

        else:
            with open("data_binary", "rb") as fp:  # Unpickling
                data_binary = pickle.load(fp)
            with open("val_binary", "rb") as fp:  # Unpickling
                val_binary = pickle.load(fp)
            with open("property_dict", "rb") as fp:  # Unpickling
                property_dict = pickle.load(fp)
            # with open("disease_dict", "rb") as fp:  # Unpickling
            #     disease_dict = pickle.load(fp)

        # healthy_idx = disease_dict['control']
        train_rows = [TrainRow(health, props) for health, *props in data_binary]
        X = np.stack([r.features.T for r in train_rows]).reshape(len(train_rows), -1)
        y = np.array([int(r.is_sick) for r in train_rows])

        val_rows = [ValRow(*p) for p in val_binary]
        val_X = np.stack([r.features.T for r in val_rows]).reshape(len(val_rows), -1)
        val_patient_ids = np.array([r.patient_id for r in val_rows])

        logger.info('transforming to X and y done')

        kwargs = dict(max_depth=10, random_state=0)
        logger.info(f'init an RF with {kwargs}')
        clf = RandomForestClassifier(**kwargs)

        logger.info(f'fitting ...')
        clf.fit(X, y)

        logger.info(f'predicting ...')
        predicted_classes = clf.predict(val_X)
        results = pd.DataFrame(np.vstack((val_patient_ids, predicted_classes)).T,
                               columns=['subject_id', 'disease'])
        logger.info(results)

        write_csv(results)

        # Close the driver connection
        driver.close()

        return 'terminal'
