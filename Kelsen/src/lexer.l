%{
#include "parser.tab.h"
#include <string.h>
%}

%option noyywrap

%%

"int"                   { return TOKEN_TYPE_INT; }
"float"                 { return TOKEN_TYPE_FLOAT; }
"string"                { return TOKEN_TYPE_STRING; }
"asset"                 { return TOKEN_TYPE_ASSET; }
"subject"               { return TOKEN_TYPE_SUBJECT; }
"clause"                { return TOKEN_TYPE_CLAUSE; }
"query"                 { return TOKEN_TYPE_QUERY; }

"AND"                   { return TOKEN_AND; }
"add"                   { return TOKEN_ADD; }
"sub"                   { return TOKEN_SUB; }
"+"                     { return TOKEN_PLUS; }
"-"                     { return TOKEN_MINUS; }
"="                     { return TOKEN_ASSIGN; }
";"                     { return TOKEN_SEMICOLON; }
","                     { return TOKEN_COMMA; }

"CR("                   { return TOKEN_CR_OPEN; }
"OB("                   { return TOKEN_OB_OPEN; }
"PR("                   { return TOKEN_PR_OPEN; }
"PVG("                  { return TOKEN_PVG_OPEN; }


"Service"               { return TOKEN_SERVICE; }
"Property"              { return TOKEN_PROPERTY; }
"NM"                    { return TOKEN_NM; }
"M"                     { return TOKEN_M; }

"PWR("                  { return TOKEN_PWR_OPEN; }
"LIAB("                 { return TOKEN_LIAB_OPEN; }
"DIS("                  { return TOKEN_DIS_OPEN; }
"IMM("                  { return TOKEN_IMM_OPEN; }

")"                     { return TOKEN_CLOSE_PAREN; }
"{"                     { return TOKEN_OPEN_BRACE; }
"}"                     { return TOKEN_CLOSE_BRACE; }

"condition:"            { return TOKEN_CONDITION; }
"consequence:"          { return TOKEN_CONSEQUENCE; }

"string_literal" { yylval.str = strdup(yytext); return STRING_LITERAL; }
"identifier" { yylval.str = strdup(yytext); return IDENTIFIER; }

\"([^"\\]|\\.)*\"       { yylval.sval = strdup(yytext); return TOKEN_STRING_LITERAL; }
[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ_][a-zA-ZáéíóúñüÁÉÍÓÚÑÜ_0-9]*  { yylval.sval = strdup(yytext); return TOKEN_IDENTIFIER; }
[0-9]+\.[0-9]+          { yylval.fval = atof(yytext); return TOKEN_FLOAT_LITERAL; }
[0-9]+                  { yylval.ival = atoi(yytext); return TOKEN_INT_LITERAL; }

[ \t\n]+                { /* Ignore whitespace */ }

.                       { printf("Error: Unknown token '%s'\n", yytext); return TOKEN_UNKNOWN; }

%%

int mywrap() {
    return 1;
}

