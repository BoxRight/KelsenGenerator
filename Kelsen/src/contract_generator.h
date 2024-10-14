#ifndef CONTRACT_GENERATOR_H
#define CONTRACT_GENERATOR_H

#include <jansson.h>

void generate_contract(const char *json_file);

const char* get_json_string_value(json_t *json_obj, const char *key);

void handle_condition(json_t *condition, const char *oferente_name);

void handle_consequence(json_t *consequence, const char *oferente_name, const char *acreedor_name);

#endif // CONTRACT_GENERATOR_H

