import numpy as np

def equalize_vectors(v1, v2):
    min_len = min(len(v1), len(v2))
    return v1[:min_len], v2[:min_len]


def getUnitVector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def calcCos(vector1, vector2):
    UnitVector1 = getUnitVector(vector1)
    UnitVector2 = getUnitVector(vector2)

    UnitVector1, UnitVector2 = equalize_vectors(UnitVector1, UnitVector2)

    similarity = np.dot(UnitVector1, UnitVector2)
    return np.clip(similarity, -1.0, 1.0)


def flattenMatrix(matrix):
    return np.matrix.flatten(matrix)

def checkSimilarity(matrix1, matrix2):
    
    similarity = calcCos(flattenMatrix(matrix1), flattenMatrix(matrix2))
    return similarity


TEST_ORIGINAL_DICT ="/home/diego/AI_Plagiarism_Detector/matrix_data/original/case-01/orig-case-01-T1.txt"
TEST_PLAG_DICT =  "/home/diego/AI_Plagiarism_Detector/matrix_data/train/case-01/plag-case-01-L3.txt"
TEST_NONPLAG_DICT =  "/home/diego/AI_Plagiarism_Detector/matrix_data/validation/case-01/nonplag-case-01-No1.txt"

with open(TEST_ORIGINAL_DICT, "r", encoding="utf-8") as f:
    test_original_matrix = f.read()

test_original_matrix = np.fromstring(test_original_matrix, sep=' ').reshape(-1, 1)

with open(TEST_PLAG_DICT, "r", encoding="utf-8") as f:
    test_plag_matrix = f.read()
test_plag_matrix = np.fromstring(test_plag_matrix, sep=' ').reshape(-1, 1)

with open(TEST_NONPLAG_DICT, "r", encoding="utf-8") as f:
    test_nonplag_matrix = f.read()
test_nonplag_matrix = np.fromstring(test_nonplag_matrix, sep=' ').reshape(-1, 1)


print("testing original matrix")

plag_similarity = checkSimilarity(test_original_matrix, test_plag_matrix)
print(f"Plagiarism similarity: {plag_similarity}")

nonplag_similarity = checkSimilarity(test_original_matrix, test_nonplag_matrix)
print(f"Non-plagiarism similarity: {nonplag_similarity}")


