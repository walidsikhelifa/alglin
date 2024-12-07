import numpy as np

def check_linear_independence(vectors):
    
    rank = np.linalg.matrix_rank(vectors)
    return rank == len(vectors)

# Exemple de vecteurs (vecteurs linéairement indépendants)
v1 = np.array([1, 2, 3])
v2 = np.array([1, 0, -1])
v3 = np.array([0, 1, 4])

# Créer une matrice avec les vecteurs comme colonnes
mat = np.column_stack((v1, v2, v3))
def gram_schmidt(A):
    """
    Effectue l'orthogonalisation de Gram-Schmidt sur la matrice A.
    
    Arguments:
    A : numpy.ndarray
        La matrice d'entrée (chaque colonne est un vecteur).
        
    Retourne:
    numpy.ndarray
        La base orthogonale des vecteurs de A.
    """
    # Nombre de vecteurs et leur dimension
    m, n = A.shape
    Q = np.zeros((m, n))  # Matrice pour stocker les vecteurs orthogonaux
    
    for i in range(n):
        v = A[:, i]  # Vecteur actuel
        
        # Soustraction des projections sur les vecteurs précédents
        for j in range(i):
            proj = np.dot(Q[:, j], v) / np.dot(Q[:, j], Q[:, j]) * Q[:, j]
            v = v - proj
        
        # Vérification si le vecteur est "quasi nul"
        if np.linalg.norm(v) < 1e-6:  # Tolérance plus large pour éviter les erreurs d'arrondi
            raise ValueError(f"Le vecteur {i+1} est nul ou trop proche de zéro, les vecteurs sont dépendants.")
        
        # Normalisation du vecteur
        Q[:, i] = v / np.linalg.norm(v)
    
    return Q


# Vérifier l'indépendance linéaire
if check_linear_independence(mat):
    print("Les vecteurs sont linéairement indépendants.")
    
    # Appliquer l'algorithme de Gram-Schmidt pour obtenir la base orthogonale
    try:
        orthogonal_basis = gram_schmidt(mat)
        print("Base orthogonale :")
        print(orthogonal_basis)
    except ValueError as e:
        print(e)
else:
    print("Les vecteurs sont linéairement dépendants.")
