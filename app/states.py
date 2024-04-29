import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import Role
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

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

    def run_bak(self):
        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"\nNeo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        logger.info(f'The neo4j_credentials are: {neo4j_credentials}')

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

    def run(self):
        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"\nNeo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")

        logger.info(f'The neo4j_credentials are: {neo4j_credentials}')
        # Driver instantiation
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # Create a driver session with defined DB
        with driver.session(database=NEO4J_DB) as session:

            query = (
                "MATCH (b:Biological_sample)-[:HAS_DISEASE]->() "
                "RETURN b.subjectid"
            )
            result = session.run(query)
            relevant_subjects = np.array([record["b.subjectid"] for record in result])
            relevant_subjects = np.unique(relevant_subjects)
            query_val = """
            MATCH (b:Biological_sample)
            WHERE NOT (b)-[:HAS_DISEASE]->()
            RETURN b.subjectid
            """
            result_val = session.run(query_val)
            val_subjects = np.array([record["b.subjectid"] for record in result_val])
            val_subjects = np.unique(val_subjects)

            data_total = []
            property_total = []
            disease_total = []
            for subject in relevant_subjects:
                query_info = (
                    "MATCH (b:Biological_sample {subjectid: $sample_id}) "
                    "OPTIONAL MATCH (b)-[:HAS_DISEASE]->(d:Disease) "
                    "OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype) "
                    "OPTIONAL MATCH (b)-[:HAS_DAMAGE]->(g:Gene) "
                    "RETURN b, "
                    "COLLECT(DISTINCT CASE WHEN EXISTS(d.id) THEN d.id ELSE d.name END) AS diseases_synonyms, "
                    "COLLECT(DISTINCT g.id) AS genes_ids, "
                    "COLLECT(DISTINCT ph.id) AS phenotypes_ids"
                )
                result_info = session.run(query_info, sample_id=subject)
                result_info = result_info.single()
                subject_property = result_info[2] + result_info[3]
                subject_health = result_info[1]
                data_total.append([subject_property, subject_health])
                property_total.extend(result_info[2] + result_info[3])
                disease_total.extend(result_info[1])

            data_val = []
            for subject in val_subjects:
                query_info = (
                    "MATCH (b:Biological_sample {subjectid: $sample_id}) "
                    "OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype) "
                    "OPTIONAL MATCH (b)-[:HAS_DAMAGE]->(g:Gene) "
                    "RETURN b, "
                    "COLLECT(DISTINCT g.id) AS genes_ids, "
                    "COLLECT(DISTINCT ph.id) AS phenotypes_ids"
                )
                result_info = session.run(query_info, sample_id=subject)
                result_info = result_info.single()
                subject_property = result_info[1] + result_info[2]
                data_val.append([subject, subject_property])
                property_total.extend(result_info[1] + result_info[2])

            property_total = np.unique(np.array(property_total))
            disease_total = np.unique(np.array(disease_total))
            property_dict = {value: index for index, value in enumerate(property_total)}
            disease_dict = {value: index for index, value in enumerate(disease_total)}
            data_binary = []
            for i in data_total:
                p_list = [property_dict[value] for value in i[0]]
                binary_p = [0] * len(property_total)
                for index in p_list:
                    binary_p[index] = 1

                d_list = [disease_dict[value] for value in i[1]]
                binary_d = [0] * len(disease_dict)
                for index in d_list:
                    binary_d[index] = 1
                data_binary.append([binary_p, binary_d])

            val_binary = []
            for i in data_val:
                p_list = [property_dict[value] for value in i[1]]
                binary_p = [0] * len(property_total)
                for index in p_list:
                    binary_p[index] = 1
                val_binary.append([i[0], binary_p])

            random.shuffle(data_binary)
            X = []
            y = []
            val = []
            subjects = []
            control_index = disease_dict["control"]
            health_status = []
            for i in data_binary:
                X.append(i[0])
                y.append(i[1])
                if i[1][control_index]:
                    health_status.append(0)
                else:
                    health_status.append(1)
            health_status = torch.tensor(health_status)

            for i in val_binary:
                val.append(i[1])
                subjects.append(i[0])

            class SimpleNN(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super(SimpleNN, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
                    self.relu = nn.ReLU()  # Activation function
                    self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second fully connected layer

                def forward(self, x):
                    x = self.fc1(x)  # Pass through first fully connected layer
                    x = self.relu(x)  # Apply ReLU activation
                    x = self.fc2(x)  # Pass through second fully connected layer
                    return x

            def get_key_from_value(dictionary, value):
                for key, val in dictionary.items():
                    if val == value:
                        return key
                return None

            def train(model, train_loader, criterion, optimizer, num_epochs):
                model.train()  # Set the model to training mode
                for epoch in range(num_epochs):
                    running_loss = 0.0
                    for inputs, targets in train_loader:
                        optimizer.zero_grad()  # Zero the gradients
                        outputs = model(inputs)  # Forward pass
                        loss = criterion(outputs, targets)  # Calculate the loss
                        loss.backward()  # Backward pass
                        optimizer.step()  # Update weights
                        running_loss += loss.item() * inputs.size(0)
                    epoch_loss = running_loss / len(train_loader.dataset)
                    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # Define the testing function
            def test(model, test_loader):
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, targets in test_loader:
                        outputs = model(inputs)  # Forward pass
                        _, predicted = torch.max(outputs, 1)  # Get the predicted class
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                    accuracy = correct / total
                    logger.info(f"Test Accuracy: {accuracy:.4f}")

            def predict(learning_rate, batch_size, num_epochs, output_dim, X, y, val, health_status, subjects):
                input_dim = len(data_binary[0][0])
                hidden_dim = 2 * len(data_binary[0][0])
                model = SimpleNN(input_dim, hidden_dim, output_dim)
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                val = torch.tensor(val, dtype=torch.float32)
                health_status = health_status.unsqueeze(1).float()
                if output_dim == 1:
                    criterion = nn.BCEWithLogitsLoss()
                    train_loader = DataLoader(TensorDataset(X, health_status), batch_size=batch_size, shuffle=True)
                else:
                    criterion = nn.CrossEntropyLoss()
                    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                train(model, train_loader, criterion, optimizer, num_epochs)
                model.eval()
                with torch.no_grad():
                    if output_dim == 1:
                        predicted_probs = torch.sigmoid(model(val))
                        predicted_classes = (predicted_probs > 0.5).squeeze().int().tolist()
                    else:
                        output_predictions = model(val)  # Forward pass
                        _, predicted_classes = torch.max(output_predictions, 1)
                        predicted_classes = predicted_classes.tolist()

                # Export DataFrame to CSV file
                if output_dim == 1:
                    data = {'subjectid': subjects, 'Disease': predicted_classes}
                    logger.info(f'len subjects is {len(subjects)}')
                    logger.info(f'len predicted_classes {len(predicted_classes)}')
                    df = pd.DataFrame(data)
                    write_csv(df, 'output_a.csv')
                else:
                    icd = []
                    for i in predicted_classes:
                        key = get_key_from_value(disease_dict, i)
                        query = """
                        MATCH (d:Disease {id: $key})
                        RETURN d, d.synonyms AS synonyms
                        """
                        result_info = session.run(query, key=key).data()
                    try:
                        synonyms = result_info[0]["synonyms"]
                        for cha in synonyms:
                            if cha[0:5]=="ICD10":
                                icd.append(cha[8])
                    except Exception:
                        icd.append('J')
                    icd = icd[0:len(subjects)]
                    logger.info(f'len subjects is {len(subjects)}')
                    logger.info(f'len icd is {len(icd)}')
                    df = pd.DataFrame(data)
                    write_csv(df, 'output_b.csv')

            learning_rate = 0.001
            batch_size = 16
            num_epochs = 10
            predict(learning_rate, batch_size, num_epochs, 1, X, y, val, health_status, subjects)
            predict(learning_rate, batch_size, num_epochs, len(data_binary[0][1]), X, y, val, health_status, subjects)

            session.close()

        return 'terminal'
