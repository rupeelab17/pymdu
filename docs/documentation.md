## Construire la documentation
Pour générer la documentation en local avec ce projet, suivez les étapes ci-dessous :

1. Cloner le projet : `git clone https://github.com/rupeelab17/pymdu.git`
2. Prévisualisez la documentation : `mkdocs serve`
3. La documentation sera accessible à l'adresse suivante : `http://127.0.0.1:8000/mdu/pymdu/site/`.
4. Ajouter du contenu dans les doctrings pour la documentation des classes et méthodes.
5. Générer le site web : `mkdocs build`.

## Formater son code avec les Doctrings : Google Style 

=== "Google style"

    ```python 
    def function_name(param1: Type1, param2: Type2, ...) -> ReturnType:
        """Brief description of the function.
        
        More detailed explanation of the function if necessary. This can span
        multiple lines as needed.
        
        Args:
           param1 (Type1): Description of param1.
           param2 (Type2): Description of param2.
           ...
        
        Returns:
           ReturnType: Description of the return value.
        
        Raises:
           ExceptionType: Explanation of the conditions under which this exception is raised.
        
        Example:
           ```python exec="false" source="tabbed-right" html="1" tabs="Source code|Plot"
           function_name(param1_value, param2_value)
           ```
        """
    ```

=== "Exemple"

    ```python
    def add_numbers(num1: int, num2: int = 5) -> int:
       """Adds two numbers together.
   
       Args:
           num1 (int): The first number to add.
           num2 (int, optional): The second number to add. Defaults to 5.
   
       Returns:
           int: The sum of num1 and num2.
   
       Example:
           ```python exec="false" source="tabbed-right" html="1" tabs="Source code|Plot"
            add_numbers(3, 2)
           ```
       """
       return num1 + num2
    ```


## Aide 

!!! note "Documentation de mkdocs"

    [https://squidfunk.github.io/mkdocs-material/reference/formatting/](https://squidfunk.github.io/mkdocs-material/reference/formatting/)

    [TIPS](https://squidfunk.github.io/mkdocs-material/reference/tooltips/#adding-abbreviations)
