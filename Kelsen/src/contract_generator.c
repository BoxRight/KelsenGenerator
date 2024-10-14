#include <stdio.h>
#include <stdlib.h>
#include <jansson.h>
#include <string.h>
#include "contract_generator.h"

// Function to extract string value from JSON object
const char* get_json_string_value(json_t *json_obj, const char *key) {
    json_t *value = json_object_get(json_obj, key);
    if (json_is_string(value)) {
        return json_string_value(value);
    }
    return NULL;
}

void handle_condition(json_t *condition, const char *oferente_name) {
    if (!condition) {
        printf("No condition found.\n");
        return;
    }

    const char *condition_type = get_json_string_value(condition, "NodeType");

    if (condition_type && strcmp(condition_type, "AST_AND") == 0) {
        json_t *left = json_object_get(condition, "Left");
        json_t *right = json_object_get(condition, "Right");
        if (left && right) {
            printf("Si el \"%s\" (OFERENTE) ", oferente_name);
            handle_condition(left, oferente_name);
            printf(" y ");
            handle_condition(right, oferente_name);
            printf(",\n");
        } else {
            printf("Left or right condition is missing in AND condition.\n");
        }
    } else if (condition_type && strcmp(condition_type, "AST_CONDITION") == 0) {
        json_t *left = json_object_get(condition, "Left");
        if (left) {
            handle_condition(left, oferente_name);
        } else {
            printf("Left part of condition is missing.\n");
        }
    } else if (condition_type && strcmp(condition_type, "AST_IDENTIFIER") == 0) {
        const char *identifier = get_json_string_value(condition, "Identifier");
        if (identifier) {
            printf("%s", identifier);
        } else {
            printf("Identifier missing in condition.\n");
        }
    } else {
        printf("Unhandled condition type: %s\n", condition_type);
    }
}




// Function to handle consequences
void handle_consequence(json_t *consequence, const char *oferente_name, const char *acreedor_name) {
    if (consequence) {
        json_t *left = json_object_get(consequence, "Left");
        if (left) {
            const char *left_identifier = get_json_string_value(left, "Identifier");
            if (left_identifier) {
                printf("entonces el \"%s\" (ACREEDOR) puede exigir al \"%s\" (OFERENTE) que %s.\n\n", acreedor_name, oferente_name, left_identifier);
            }
        }
    }
}

// Function to generate the contract
void generate_contract(const char *json_file) {
    json_error_t error;
    json_t *root = json_load_file(json_file, 0, &error);
    if (!root) {
        fprintf(stderr, "Error loading JSON: %s\n", error.text);
        return;
    }

    // Extract subjects
    const char *oferente_name = NULL;
    const char *oferente_address = NULL;
    const char *acreedor_name = NULL;
    const char *acreedor_address = NULL;

    json_t *node = root;
    while (node) {
        const char *node_type = get_json_string_value(node, "NodeType");
        if (node_type && strcmp(node_type, "AST_DECLARATION") == 0) {
            json_t *identifier_obj = json_object_get(node, "Identifier");
            if (identifier_obj) {
                const char *identifier = get_json_string_value(identifier_obj, "Identifier");
                if (identifier && strcmp(identifier, "OFERENTE") == 0) {
                    json_t *expression = json_object_get(node, "Expression");
                    if (expression) {
                        oferente_name = get_json_string_value(json_object_get(expression, "Description1"), "String");
                        oferente_address = get_json_string_value(json_object_get(expression, "Description2"), "String");
                    }
                } else if (identifier && strcmp(identifier, "ACREEDOR") == 0) {
                    json_t *expression = json_object_get(node, "Expression");
                    if (expression) {
                        acreedor_name = get_json_string_value(json_object_get(expression, "Description1"), "String");
                        acreedor_address = get_json_string_value(json_object_get(expression, "Description2"), "String");
                    }
                }
            }
        }
        node = json_object_get(node, "Next");
    }

    // Print contract header
    printf("Contrato\n\nPARTES:\n\nOFERENTE:\n\n");
    if (oferente_name && oferente_address) {
        printf("    Nombre: \"%s\"\n    Dirección: \"%s\"\n    Para oír y recibir notificaciones\n\n", oferente_name, oferente_address);
    }
    printf("ACREEDOR:\n\n");
    if (acreedor_name && acreedor_address) {
        printf("    Nombre: \"%s\"\n    Dirección: \"%s\"\n    Para oír y recibir notificaciones\n\n", acreedor_name, acreedor_address);
    }

    // Extract assets
    char *conductas[10];
    int conductas_index = 0;
    char *bienes[10];
    int bienes_index = 0;

    node = root;
    while (node) {
        const char *node_type = get_json_string_value(node, "NodeType");
        if (node_type && strcmp(node_type, "AST_DECLARATION") == 0) {
            json_t *identifier_obj = json_object_get(node, "Identifier");
            if (identifier_obj) {
                const char *identifier = get_json_string_value(identifier_obj, "Identifier");
                if (identifier && (strcmp(identifier, "Ofrecer") == 0 || strcmp(identifier, "Sostener") == 0 || strcmp(identifier, "Vender") == 0)) {
                    json_t *expression = json_object_get(node, "Expression");
                    if (expression) {
                        const char *subject1 = get_json_string_value(json_object_get(expression, "Subject1"), "Identifier");
                        const char *description = get_json_string_value(json_object_get(expression, "Description"), "Identifier");
                        const char *subject2 = get_json_string_value(json_object_get(expression, "Subject2"), "Identifier");
                        const char *type = get_json_string_value(json_object_get(expression, "Type"), "Type");

                        char *asset_info = malloc(256);
                        if (strcmp(type, "Service") == 0) {
                            snprintf(asset_info, 256, "    El OFERENTE hace una oferta al ACREEDOR respecto a la conducta del activo que el OFERENTE %s al ACREEDOR.\n", description);
                            conductas[conductas_index++] = asset_info;
                        } else if (strcmp(type, "Property") == 0) {
                            snprintf(asset_info, 256, "    El OFERENTE vende un bien inmueble al ACREEDOR.\n");
                            bienes[bienes_index++] = asset_info;
                        }
                    }
                }
            }
        }
        node = json_object_get(node, "Next");
    }

    // Print assets
    printf("OBJETO DEL CONTRATO:\n\n    Conducta del Activo:\n");
    for (int i = 0; i < conductas_index; i++) {
        printf("%s", conductas[i]);
        free(conductas[i]);
    }
    printf("\n    Bien Inmueble:\n");
    for (int i = 0; i < bienes_index; i++) {
        printf("%s", bienes[i]);
        free(bienes[i]);
    }

    printf("\nCLÁUSULAS:\n\n");

    // Extract and print clauses
    int clause_number = 1;
    node = root;
    while (node) {
        const char *node_type = get_json_string_value(node, "NodeType");
        if (node_type && strcmp(node_type, "AST_DECLARATION") == 0) {
            json_t *identifier_obj = json_object_get(node, "Identifier");
            if (identifier_obj) {
                const char *identifier = get_json_string_value(identifier_obj, "Identifier");
                if (identifier && (strcmp(identifier, "a") == 0 || strcmp(identifier, "b") == 0)) {
                    json_t *expression = json_object_get(node, "Expression");
                    if (expression) {
                        json_t *condition = json_object_get(expression, "Condition");
                        json_t *consequence = json_object_get(expression, "Consequence");

                        printf("CLÁUSULA %d:\n", clause_number++);
                        printf("Si el ");
                        handle_condition(condition, oferente_name);
                        handle_consequence(consequence, oferente_name, acreedor_name);
                    }
                }
            }
        }
        node = json_object_get(node, "Next");
    }

    json_decref(root);
}


