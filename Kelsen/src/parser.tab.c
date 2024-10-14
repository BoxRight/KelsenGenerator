/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "parser.y"

#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "symbol_table.h"

ASTNode *root;  // Define the root variable
extern SymbolTable *symbol_table;

extern int yylex(void);
extern int yylineno;
extern char *yytext;
int check_declaration_types(ASTNode *declaration);
int check_expression_types(ASTNode *expression, VarType expected_type);



void yyerror(const char *s) {
    fprintf(stderr, "Error: %s at line %d, near token '%s'\n", s, yylineno, yytext);
}

#define YYDEBUG 1

#line 95 "parser.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "parser.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_TOKEN_INT_LITERAL = 3,          /* TOKEN_INT_LITERAL  */
  YYSYMBOL_TOKEN_FLOAT_LITERAL = 4,        /* TOKEN_FLOAT_LITERAL  */
  YYSYMBOL_TOKEN_STRING_LITERAL = 5,       /* TOKEN_STRING_LITERAL  */
  YYSYMBOL_TOKEN_IDENTIFIER = 6,           /* TOKEN_IDENTIFIER  */
  YYSYMBOL_TOKEN_TYPE_INT = 7,             /* TOKEN_TYPE_INT  */
  YYSYMBOL_TOKEN_TYPE_FLOAT = 8,           /* TOKEN_TYPE_FLOAT  */
  YYSYMBOL_TOKEN_TYPE_STRING = 9,          /* TOKEN_TYPE_STRING  */
  YYSYMBOL_TOKEN_TYPE_ASSET = 10,          /* TOKEN_TYPE_ASSET  */
  YYSYMBOL_TOKEN_TYPE_SUBJECT = 11,        /* TOKEN_TYPE_SUBJECT  */
  YYSYMBOL_TOKEN_TYPE_CLAUSE = 12,         /* TOKEN_TYPE_CLAUSE  */
  YYSYMBOL_TOKEN_TYPE_QUERY = 13,          /* TOKEN_TYPE_QUERY  */
  YYSYMBOL_TOKEN_ADD = 14,                 /* TOKEN_ADD  */
  YYSYMBOL_TOKEN_SUB = 15,                 /* TOKEN_SUB  */
  YYSYMBOL_TOKEN_PLUS = 16,                /* TOKEN_PLUS  */
  YYSYMBOL_TOKEN_MINUS = 17,               /* TOKEN_MINUS  */
  YYSYMBOL_TOKEN_ASSIGN = 18,              /* TOKEN_ASSIGN  */
  YYSYMBOL_TOKEN_SEMICOLON = 19,           /* TOKEN_SEMICOLON  */
  YYSYMBOL_TOKEN_COMMA = 20,               /* TOKEN_COMMA  */
  YYSYMBOL_TOKEN_CR_OPEN = 21,             /* TOKEN_CR_OPEN  */
  YYSYMBOL_TOKEN_OB_OPEN = 22,             /* TOKEN_OB_OPEN  */
  YYSYMBOL_TOKEN_PR_OPEN = 23,             /* TOKEN_PR_OPEN  */
  YYSYMBOL_TOKEN_PVG_OPEN = 24,            /* TOKEN_PVG_OPEN  */
  YYSYMBOL_TOKEN_CLOSE_PAREN = 25,         /* TOKEN_CLOSE_PAREN  */
  YYSYMBOL_TOKEN_OPEN_BRACE = 26,          /* TOKEN_OPEN_BRACE  */
  YYSYMBOL_TOKEN_CLOSE_BRACE = 27,         /* TOKEN_CLOSE_BRACE  */
  YYSYMBOL_TOKEN_CONDITION = 28,           /* TOKEN_CONDITION  */
  YYSYMBOL_TOKEN_CONSEQUENCE = 29,         /* TOKEN_CONSEQUENCE  */
  YYSYMBOL_TOKEN_AND = 30,                 /* TOKEN_AND  */
  YYSYMBOL_TOKEN_PWR_OPEN = 31,            /* TOKEN_PWR_OPEN  */
  YYSYMBOL_TOKEN_LIAB_OPEN = 32,           /* TOKEN_LIAB_OPEN  */
  YYSYMBOL_TOKEN_DIS_OPEN = 33,            /* TOKEN_DIS_OPEN  */
  YYSYMBOL_TOKEN_IMM_OPEN = 34,            /* TOKEN_IMM_OPEN  */
  YYSYMBOL_TOKEN_UNKNOWN = 35,             /* TOKEN_UNKNOWN  */
  YYSYMBOL_TOKEN_SERVICE = 36,             /* TOKEN_SERVICE  */
  YYSYMBOL_TOKEN_PROPERTY = 37,            /* TOKEN_PROPERTY  */
  YYSYMBOL_TOKEN_NM = 38,                  /* TOKEN_NM  */
  YYSYMBOL_TOKEN_M = 39,                   /* TOKEN_M  */
  YYSYMBOL_YYACCEPT = 40,                  /* $accept  */
  YYSYMBOL_program = 41,                   /* program  */
  YYSYMBOL_declarations = 42,              /* declarations  */
  YYSYMBOL_declaration = 43,               /* declaration  */
  YYSYMBOL_type = 44,                      /* type  */
  YYSYMBOL_identifier = 45,                /* identifier  */
  YYSYMBOL_expression = 46,                /* expression  */
  YYSYMBOL_numeric_expression = 47,        /* numeric_expression  */
  YYSYMBOL_additive_expression = 48,       /* additive_expression  */
  YYSYMBOL_string_expression = 49,         /* string_expression  */
  YYSYMBOL_asset_expression = 50,          /* asset_expression  */
  YYSYMBOL_subject_expression = 51,        /* subject_expression  */
  YYSYMBOL_claim_right = 52,               /* claim_right  */
  YYSYMBOL_obligation = 53,                /* obligation  */
  YYSYMBOL_prohibition = 54,               /* prohibition  */
  YYSYMBOL_privilege = 55,                 /* privilege  */
  YYSYMBOL_condition_expression = 56,      /* condition_expression  */
  YYSYMBOL_clause_expression = 57,         /* clause_expression  */
  YYSYMBOL_power = 58,                     /* power  */
  YYSYMBOL_liability = 59,                 /* liability  */
  YYSYMBOL_disability = 60,                /* disability  */
  YYSYMBOL_immunity = 61,                  /* immunity  */
  YYSYMBOL_query_expression = 62,          /* query_expression  */
  YYSYMBOL_asset_type = 63,                /* asset_type  */
  YYSYMBOL_asset_subtype = 64              /* asset_subtype  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  12
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   161

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  40
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  25
/* YYNRULES -- Number of rules.  */
#define YYNRULES  75
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  162

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   294


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,    58,    58,    62,    63,    67,    76,    85,    86,    87,
      88,    89,    90,    91,    97,   101,   102,   103,   104,   105,
     106,   107,   110,   111,   115,   116,   120,   125,   141,   160,
     182,   191,   203,   211,   222,   230,   241,   249,   259,   267,
     275,   283,   297,   306,   314,   322,   330,   338,   346,   354,
     366,   374,   382,   390,   401,   409,   417,   425,   436,   444,
     452,   460,   471,   479,   487,   495,   505,   515,   525,   534,
     546,   547,   551,   552,   553,   554
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "TOKEN_INT_LITERAL",
  "TOKEN_FLOAT_LITERAL", "TOKEN_STRING_LITERAL", "TOKEN_IDENTIFIER",
  "TOKEN_TYPE_INT", "TOKEN_TYPE_FLOAT", "TOKEN_TYPE_STRING",
  "TOKEN_TYPE_ASSET", "TOKEN_TYPE_SUBJECT", "TOKEN_TYPE_CLAUSE",
  "TOKEN_TYPE_QUERY", "TOKEN_ADD", "TOKEN_SUB", "TOKEN_PLUS",
  "TOKEN_MINUS", "TOKEN_ASSIGN", "TOKEN_SEMICOLON", "TOKEN_COMMA",
  "TOKEN_CR_OPEN", "TOKEN_OB_OPEN", "TOKEN_PR_OPEN", "TOKEN_PVG_OPEN",
  "TOKEN_CLOSE_PAREN", "TOKEN_OPEN_BRACE", "TOKEN_CLOSE_BRACE",
  "TOKEN_CONDITION", "TOKEN_CONSEQUENCE", "TOKEN_AND", "TOKEN_PWR_OPEN",
  "TOKEN_LIAB_OPEN", "TOKEN_DIS_OPEN", "TOKEN_IMM_OPEN", "TOKEN_UNKNOWN",
  "TOKEN_SERVICE", "TOKEN_PROPERTY", "TOKEN_NM", "TOKEN_M", "$accept",
  "program", "declarations", "declaration", "type", "identifier",
  "expression", "numeric_expression", "additive_expression",
  "string_expression", "asset_expression", "subject_expression",
  "claim_right", "obligation", "prohibition", "privilege",
  "condition_expression", "clause_expression", "power", "liability",
  "disability", "immunity", "query_expression", "asset_type",
  "asset_subtype", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-82)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      72,   -82,   -82,   -82,   -82,   -82,   -82,   -82,     7,    72,
     -82,    16,   -82,   -82,   -82,    49,    39,   -82,   -82,   -82,
     -82,    11,    16,    16,    16,    16,   -82,   -82,    12,    76,
     -82,    19,   -82,   -82,   -82,   -82,   -82,   -82,   -82,   -82,
      25,    44,    47,    78,    79,    80,    81,    82,   -82,    89,
      89,    64,    -5,    16,    58,    -8,    16,    16,    16,    16,
     -82,   -82,    83,   -82,   -82,   -82,   -82,    84,   -82,   -82,
      11,    11,    11,    11,    85,    86,    87,    88,    90,    92,
      93,    94,    98,   102,   103,   104,    89,    91,   101,   105,
     106,   107,   108,   109,   110,   111,   -82,   -82,   -82,   -82,
     -82,   -82,   -82,   -82,    -3,    -3,    -3,    -3,   118,   119,
      19,   120,   -82,   -82,   -82,   -82,   -82,   -82,   -82,   -82,
     116,   117,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,   132,   133,   134,    64,    16,    64,   -82,
     -82,   -82,   -82,   -82,   -82,   -82,   -82,   -82,   -82,   -82,
     -82,   -82,   -82,   -82,   -82,   -82,   140,   141,    16,    64,
     -82,   -82
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     7,     8,     9,    10,    11,    12,    13,     0,     2,
       3,     0,     1,     4,    14,     0,     0,     6,    22,    23,
      26,     0,     0,     0,     0,     0,    70,    71,     0,    15,
      16,    19,    17,    18,    20,    66,    67,    68,    69,    21,
       0,    39,    38,     0,     0,     0,     0,     0,     5,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      24,    25,     0,    74,    75,    72,    73,     0,    41,    40,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    42,    43,    44,    45,
      46,    47,    48,    49,     0,     0,     0,     0,     0,     0,
       0,     0,    31,    30,    33,    32,    35,    34,    37,    36,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    62,    63,    64,    65,    29,     0,     0,     0,     0,
      28,    27
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -82,   -82,   -82,    96,   -82,   -21,   -82,   -40,   -82,   -11,
     -13,   -81,   -77,   -51,   -43,   -18,   -82,   -82,    51,    52,
      53,    54,   -82,   -82,   -82
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
       0,     8,     9,    10,    11,    15,    28,    29,    30,   110,
      32,    33,    74,    75,    76,    77,    43,    34,    35,    36,
      37,    38,    39,    40,    67
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      41,    44,    45,    46,    47,    31,   111,    12,    42,    60,
      61,    63,    64,    70,    71,    72,    73,    14,    70,    71,
      72,    73,    14,    22,    23,    24,    25,   120,   124,   128,
     132,    48,    68,    65,    66,    82,    83,    84,    85,    51,
      62,    69,    18,    19,    20,    52,   108,    26,    27,    88,
      90,    92,    94,   121,   125,   129,   133,    89,    91,    93,
      95,   122,   126,   130,   134,    21,   109,    16,    17,    20,
      22,    23,    24,    25,    53,    26,    27,    54,   161,     1,
       2,     3,     4,     5,     6,     7,   123,   127,   131,   135,
      49,    50,    18,    19,    26,    27,    20,    14,    55,    56,
      57,    58,    59,    86,    87,    13,    78,    79,    80,    81,
       0,     0,    96,    97,    98,    99,   156,   100,   104,   101,
     102,   103,   105,   106,   107,   155,   112,   157,     0,     0,
     113,   114,   115,   116,   117,   118,   119,   160,   136,   137,
     138,   139,   140,     0,     0,     0,   141,   142,   143,   144,
     145,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     158,   159
};

static const yytype_int16 yycheck[] =
{
      21,    22,    23,    24,    25,    16,    87,     0,    21,    49,
      50,    16,    17,    21,    22,    23,    24,     6,    21,    22,
      23,    24,     6,    31,    32,    33,    34,   104,   105,   106,
     107,    19,    53,    38,    39,    56,    57,    58,    59,    20,
      51,    54,     3,     4,     5,    20,    86,    36,    37,    70,
      71,    72,    73,   104,   105,   106,   107,    70,    71,    72,
      73,   104,   105,   106,   107,    26,    87,    18,    19,     5,
      31,    32,    33,    34,    30,    36,    37,    30,   159,     7,
       8,     9,    10,    11,    12,    13,   104,   105,   106,   107,
      14,    15,     3,     4,    36,    37,     5,     6,    20,    20,
      20,    20,    20,    20,    20,     9,    55,    55,    55,    55,
      -1,    -1,    27,    27,    27,    27,   137,    27,    20,    27,
      27,    27,    20,    20,    20,   136,    25,   138,    -1,    -1,
      25,    25,    25,    25,    25,    25,    25,   158,    20,    20,
      20,    25,    25,    -1,    -1,    -1,    25,    25,    25,    25,
      25,    25,    25,    25,    25,    25,    25,    25,    25,    25,
      20,    20
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     7,     8,     9,    10,    11,    12,    13,    41,    42,
      43,    44,     0,    43,     6,    45,    18,    19,     3,     4,
       5,    26,    31,    32,    33,    34,    36,    37,    46,    47,
      48,    49,    50,    51,    57,    58,    59,    60,    61,    62,
      63,    45,    50,    56,    45,    45,    45,    45,    19,    14,
      15,    20,    20,    30,    30,    20,    20,    20,    20,    20,
      47,    47,    49,    16,    17,    38,    39,    64,    45,    50,
      21,    22,    23,    24,    52,    53,    54,    55,    58,    59,
      60,    61,    45,    45,    45,    45,    20,    20,    45,    50,
      45,    50,    45,    50,    45,    50,    27,    27,    27,    27,
      27,    27,    27,    27,    20,    20,    20,    20,    47,    45,
      49,    51,    25,    25,    25,    25,    25,    25,    25,    25,
      52,    53,    54,    55,    52,    53,    54,    55,    52,    53,
      54,    55,    52,    53,    54,    55,    20,    20,    20,    25,
      25,    25,    25,    25,    25,    25,    25,    25,    25,    25,
      25,    25,    25,    25,    25,    49,    45,    49,    20,    20,
      45,    51
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    40,    41,    42,    42,    43,    43,    44,    44,    44,
      44,    44,    44,    44,    45,    46,    46,    46,    46,    46,
      46,    46,    47,    47,    48,    48,    49,    50,    50,    51,
      52,    52,    53,    53,    54,    54,    55,    55,    56,    56,
      56,    56,    57,    57,    57,    57,    57,    57,    57,    57,
      58,    58,    58,    58,    59,    59,    59,    59,    60,    60,
      60,    60,    61,    61,    61,    61,    62,    62,    62,    62,
      63,    63,    64,    64,    64,    64
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     2,     5,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     3,     1,     9,     9,     7,
       3,     3,     3,     3,     3,     3,     3,     3,     1,     1,
       3,     3,     5,     5,     5,     5,     5,     5,     5,     5,
       7,     7,     7,     7,     7,     7,     7,     7,     7,     7,
       7,     7,     7,     7,     7,     7,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* program: declarations  */
#line 58 "parser.y"
                   { root = (yyvsp[0].node); }
#line 1257 "parser.tab.c"
    break;

  case 3: /* declarations: declaration  */
#line 62 "parser.y"
                  { (yyval.node) = (yyvsp[0].node); }
#line 1263 "parser.tab.c"
    break;

  case 4: /* declarations: declarations declaration  */
#line 63 "parser.y"
                               { (yyval.node) = create_declarations((yyvsp[-1].node), (yyvsp[0].node)); }
#line 1269 "parser.tab.c"
    break;

  case 5: /* declaration: type identifier TOKEN_ASSIGN expression TOKEN_SEMICOLON  */
#line 67 "parser.y"
                                                              {
          (yyval.node) = create_declaration((yyvsp[-4].node), (yyvsp[-3].node), (yyvsp[-1].node));
          if (!insert_symbol(symbol_table, (yyvsp[-3].node)->data.sval, (yyvsp[-4].node)->var_type)) {
              yyerror("Symbol insertion failed");
          }
          if (!check_declaration_types((yyval.node))) {
              yyerror("Type mismatch in declaration");
          }
      }
#line 1283 "parser.tab.c"
    break;

  case 6: /* declaration: type identifier TOKEN_SEMICOLON  */
#line 76 "parser.y"
                                      {
          (yyval.node) = create_declaration((yyvsp[-2].node), (yyvsp[-1].node), NULL);
          if (!insert_symbol(symbol_table, (yyvsp[-1].node)->data.sval, (yyvsp[-2].node)->var_type)) {
              yyerror("Symbol insertion failed");
          }
      }
#line 1294 "parser.tab.c"
    break;

  case 7: /* type: TOKEN_TYPE_INT  */
#line 85 "parser.y"
                     { (yyval.node) = create_type_node("int"); }
#line 1300 "parser.tab.c"
    break;

  case 8: /* type: TOKEN_TYPE_FLOAT  */
#line 86 "parser.y"
                       { (yyval.node) = create_type_node("float"); }
#line 1306 "parser.tab.c"
    break;

  case 9: /* type: TOKEN_TYPE_STRING  */
#line 87 "parser.y"
                        { (yyval.node) = create_type_node("string"); }
#line 1312 "parser.tab.c"
    break;

  case 10: /* type: TOKEN_TYPE_ASSET  */
#line 88 "parser.y"
                       { (yyval.node) = create_type_node("asset"); }
#line 1318 "parser.tab.c"
    break;

  case 11: /* type: TOKEN_TYPE_SUBJECT  */
#line 89 "parser.y"
                         { (yyval.node) = create_type_node("subject"); }
#line 1324 "parser.tab.c"
    break;

  case 12: /* type: TOKEN_TYPE_CLAUSE  */
#line 90 "parser.y"
                        { (yyval.node) = create_type_node("clause"); }
#line 1330 "parser.tab.c"
    break;

  case 13: /* type: TOKEN_TYPE_QUERY  */
#line 91 "parser.y"
                       { (yyval.node) = create_type_node("query"); }
#line 1336 "parser.tab.c"
    break;

  case 14: /* identifier: TOKEN_IDENTIFIER  */
#line 97 "parser.y"
                       { (yyval.node) = create_identifier((yyvsp[0].sval)); }
#line 1342 "parser.tab.c"
    break;

  case 15: /* expression: numeric_expression  */
#line 101 "parser.y"
                         { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), (yyvsp[0].node)->var_type)) yyerror("Type mismatch in numeric expression"); }
#line 1348 "parser.tab.c"
    break;

  case 16: /* expression: additive_expression  */
#line 102 "parser.y"
                          { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), (yyvsp[0].node)->var_type)) yyerror("Type mismatch in additive expression"); }
#line 1354 "parser.tab.c"
    break;

  case 17: /* expression: asset_expression  */
#line 103 "parser.y"
                       { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), TYPE_ASSET)) yyerror("Type mismatch in asset expression"); }
#line 1360 "parser.tab.c"
    break;

  case 18: /* expression: subject_expression  */
#line 104 "parser.y"
                         { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), TYPE_SUBJECT)) yyerror("Type mismatch in subject expression"); }
#line 1366 "parser.tab.c"
    break;

  case 19: /* expression: string_expression  */
#line 105 "parser.y"
                        { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), TYPE_STRING)) yyerror("Type mismatch in string expression"); }
#line 1372 "parser.tab.c"
    break;

  case 20: /* expression: clause_expression  */
#line 106 "parser.y"
                        { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), TYPE_CLAUSE)) yyerror("Type mismatch in clause expression"); }
#line 1378 "parser.tab.c"
    break;

  case 21: /* expression: query_expression  */
#line 107 "parser.y"
                       { (yyval.node) = (yyvsp[0].node); if (!check_expression_types((yyval.node), TYPE_QUERY)) yyerror("Type mismatch in query expression"); }
#line 1384 "parser.tab.c"
    break;

  case 22: /* numeric_expression: TOKEN_INT_LITERAL  */
#line 110 "parser.y"
                        { (yyval.node) = create_number((yyvsp[0].ival)); }
#line 1390 "parser.tab.c"
    break;

  case 23: /* numeric_expression: TOKEN_FLOAT_LITERAL  */
#line 111 "parser.y"
                          { (yyval.node) = create_float((yyvsp[0].fval)); }
#line 1396 "parser.tab.c"
    break;

  case 24: /* additive_expression: numeric_expression TOKEN_ADD numeric_expression  */
#line 115 "parser.y"
                                                      { (yyval.node) = create_binary_op(AST_ADD, (yyvsp[-2].node), (yyvsp[0].node)); if (!check_expression_types((yyval.node), TYPE_INT)) yyerror("Type error in additive expression"); }
#line 1402 "parser.tab.c"
    break;

  case 25: /* additive_expression: numeric_expression TOKEN_SUB numeric_expression  */
#line 116 "parser.y"
                                                      { (yyval.node) = create_binary_op(AST_SUB, (yyvsp[-2].node), (yyvsp[0].node)); if (!check_expression_types((yyval.node), TYPE_INT)) yyerror("Type error in additive expression"); }
#line 1408 "parser.tab.c"
    break;

  case 26: /* string_expression: TOKEN_STRING_LITERAL  */
#line 120 "parser.y"
                           { (yyval.node) = create_string((yyvsp[0].sval)); }
#line 1414 "parser.tab.c"
    break;

  case 27: /* asset_expression: asset_type TOKEN_COMMA asset_subtype TOKEN_COMMA subject_expression TOKEN_COMMA string_expression TOKEN_COMMA subject_expression  */
#line 125 "parser.y"
                                                                                                                                       {
          // Check the types of each component
          if (!check_expression_types((yyvsp[-4].node), TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the third part in asset expression");
          } else if (!check_expression_types((yyvsp[-2].node), TYPE_STRING)) {
              yyerror("Type error: Expected string type for the fourth part in asset expression");
          } else if (!check_expression_types((yyvsp[0].node), TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the fifth part in asset expression");
          } else {
              // If all checks pass, create the AST node
        (yyval.node) = create_asset_expression((yyvsp[-8].node), (yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node));
            if (!check_expression_types((yyval.node), TYPE_ASSET)) {
                yyerror("Type error in asset expression");
              }
          }
      }
#line 1435 "parser.tab.c"
    break;

  case 28: /* asset_expression: asset_type TOKEN_COMMA asset_subtype TOKEN_COMMA identifier TOKEN_COMMA identifier TOKEN_COMMA identifier  */
#line 141 "parser.y"
                                                                                                                {
          // Check the types of each component
          if (!check_expression_types((yyvsp[-4].node), TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the third part in asset expression");
          } else if (!check_expression_types((yyvsp[-2].node), TYPE_STRING)) {
              yyerror("Type error: Expected string type for the fourth part in asset expression");
          } else if (!check_expression_types((yyvsp[0].node), TYPE_SUBJECT)) {
              yyerror("Type error: Expected subject type for the fifth part in asset expression");
          } else {
              // If all checks pass, create the AST node
        (yyval.node) = create_asset_expression((yyvsp[-8].node), (yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node), (yyvsp[0].node));
            if (!check_expression_types((yyval.node), TYPE_ASSET)) {
                yyerror("Type error in asset expression");
              }
          }
      }
#line 1456 "parser.tab.c"
    break;

  case 29: /* subject_expression: string_expression TOKEN_COMMA string_expression TOKEN_COMMA numeric_expression TOKEN_COMMA string_expression  */
#line 160 "parser.y"
                                                                                                                   {
          // Check the types of each component
          if (!check_expression_types((yyvsp[-6].node), TYPE_STRING)) {
              yyerror("Type error: Expected string type for the first string expression in subject expression");
          } else if (!check_expression_types((yyvsp[-4].node), TYPE_STRING)) {
              yyerror("Type error: Expected string type for the second string expression in subject expression");
          } else if (!check_expression_types((yyvsp[-2].node), TYPE_INT)) {
              yyerror("Type error: Expected int type for the integer literal in subject expression");
          } else if (!check_expression_types((yyvsp[0].node), TYPE_STRING)) {
              yyerror("Type error: Expected string type for the third string expression in subject expression");
          } else {
              // If all checks pass, create the AST node
              (yyval.node) = create_subject_expression((yyvsp[-6].node), (yyvsp[-4].node), (yyvsp[-2].node)->data.ival, (yyvsp[0].node));
              if (!check_expression_types((yyval.node), TYPE_SUBJECT)) {
                  yyerror("Type error in subject expression");
              }
          }
      }
#line 1479 "parser.tab.c"
    break;

  case 30: /* claim_right: TOKEN_CR_OPEN asset_expression TOKEN_CLOSE_PAREN  */
#line 182 "parser.y"
                                                       { 

          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in claim-right expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_CR, (yyvsp[-1].node)); 

          }
      }
#line 1493 "parser.tab.c"
    break;

  case 31: /* claim_right: TOKEN_CR_OPEN identifier TOKEN_CLOSE_PAREN  */
#line 191 "parser.y"
                                                 { 

          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in claim-right expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_CR, (yyvsp[-1].node));

          }
      }
#line 1507 "parser.tab.c"
    break;

  case 32: /* obligation: TOKEN_OB_OPEN asset_expression TOKEN_CLOSE_PAREN  */
#line 203 "parser.y"
                                                       { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in obligation expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_OB, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1520 "parser.tab.c"
    break;

  case 33: /* obligation: TOKEN_OB_OPEN identifier TOKEN_CLOSE_PAREN  */
#line 211 "parser.y"
                                                 { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in obligation expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_OB, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1533 "parser.tab.c"
    break;

  case 34: /* prohibition: TOKEN_PR_OPEN asset_expression TOKEN_CLOSE_PAREN  */
#line 222 "parser.y"
                                                       { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in prohibition expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_PR, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1546 "parser.tab.c"
    break;

  case 35: /* prohibition: TOKEN_PR_OPEN identifier TOKEN_CLOSE_PAREN  */
#line 230 "parser.y"
                                                 { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in prohibition expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_PR, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1559 "parser.tab.c"
    break;

  case 36: /* privilege: TOKEN_PVG_OPEN asset_expression TOKEN_CLOSE_PAREN  */
#line 241 "parser.y"
                                                        { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in privilege expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_PVG, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1572 "parser.tab.c"
    break;

  case 37: /* privilege: TOKEN_PVG_OPEN identifier TOKEN_CLOSE_PAREN  */
#line 249 "parser.y"
                                                  { 
          if (!check_expression_types((yyvsp[-1].node), TYPE_ASSET)) {
              yyerror("Type error in privilege expression: expected asset");
          } else {
              (yyval.node) = create_legal(AST_PVG, (yyvsp[-1].node)); 
              (yyval.node)->var_type = TYPE_LEGAL;
          }
      }
#line 1585 "parser.tab.c"
    break;

  case 38: /* condition_expression: asset_expression  */
#line 259 "parser.y"
                       { 
          if (!check_expression_types((yyvsp[0].node), TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              (yyval.node) = create_condition((yyvsp[0].node)); 
              (yyval.node)->var_type = TYPE_CONDITION;
          }
      }
#line 1598 "parser.tab.c"
    break;

  case 39: /* condition_expression: identifier  */
#line 267 "parser.y"
                 { 
          if (!check_expression_types((yyvsp[0].node), TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              (yyval.node) = create_condition((yyvsp[0].node)); 
              (yyval.node)->var_type = TYPE_CONDITION;
          }
      }
#line 1611 "parser.tab.c"
    break;

  case 40: /* condition_expression: asset_expression TOKEN_AND asset_expression  */
#line 275 "parser.y"
                                                  { 
          if (!check_expression_types((yyvsp[-2].node), TYPE_ASSET) || !check_expression_types((yyvsp[0].node), TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              (yyval.node) = create_condition(create_binary_op(AST_AND, (yyvsp[-2].node), (yyvsp[0].node))); 
              (yyval.node)->var_type = TYPE_CONDITION;
          }
      }
#line 1624 "parser.tab.c"
    break;

  case 41: /* condition_expression: identifier TOKEN_AND identifier  */
#line 283 "parser.y"
                                      { 
          if (!check_expression_types((yyvsp[-2].node), TYPE_ASSET) || !check_expression_types((yyvsp[0].node), TYPE_ASSET)) {
              yyerror("Type error in condition expression: expected asset");
          } else {
              (yyval.node) = create_condition(create_binary_op(AST_AND, (yyvsp[-2].node), (yyvsp[0].node))); 
              (yyval.node)->var_type = TYPE_CONDITION;
          }
      }
#line 1637 "parser.tab.c"
    break;

  case 42: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA claim_right TOKEN_CLOSE_BRACE  */
#line 297 "parser.y"
                                                                                    { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_CLAIM_RIGHT)) {
            yyerror("Type error in clause expression");
            YYERROR;
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1651 "parser.tab.c"
    break;

  case 43: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA obligation TOKEN_CLOSE_BRACE  */
#line 306 "parser.y"
                                                                                     { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_OBLIGATION)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1664 "parser.tab.c"
    break;

  case 44: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA prohibition TOKEN_CLOSE_BRACE  */
#line 314 "parser.y"
                                                                                      { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_PROHIBITION)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1677 "parser.tab.c"
    break;

  case 45: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA privilege TOKEN_CLOSE_BRACE  */
#line 322 "parser.y"
                                                                                    { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_PRIVILEGE)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1690 "parser.tab.c"
    break;

  case 46: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA power TOKEN_CLOSE_BRACE  */
#line 330 "parser.y"
                                                                                { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_POWER)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1703 "parser.tab.c"
    break;

  case 47: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA liability TOKEN_CLOSE_BRACE  */
#line 338 "parser.y"
                                                                                    { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_LIABILITY)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1716 "parser.tab.c"
    break;

  case 48: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA disability TOKEN_CLOSE_BRACE  */
#line 346 "parser.y"
                                                                                     { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_DISABILITY)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1729 "parser.tab.c"
    break;

  case 49: /* clause_expression: TOKEN_OPEN_BRACE condition_expression TOKEN_COMMA immunity TOKEN_CLOSE_BRACE  */
#line 354 "parser.y"
                                                                                   { 
        if (!check_expression_types((yyvsp[-3].node), TYPE_CONDITION) || !check_expression_types((yyvsp[-1].node), TYPE_IMMUNITY)) {
            yyerror("Type error in clause expression");
        } else {
            (yyval.node) = create_clause_expression((yyvsp[-3].node), (yyvsp[-1].node)); 
            (yyval.node)->var_type = TYPE_CLAUSE;
        }
    }
#line 1742 "parser.tab.c"
    break;

  case 50: /* power: TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN  */
#line 366 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in power expression");
          } else {
              (yyval.node) = create_legal(AST_PWR, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_POWER;
          }
      }
#line 1755 "parser.tab.c"
    break;

  case 51: /* power: TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN  */
#line 374 "parser.y"
                                                                                                { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_OBLIGATION)) {
              yyerror("Type error in power expression");
          } else {
              (yyval.node) = create_legal(AST_PWR, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_POWER;
          }
      }
#line 1768 "parser.tab.c"
    break;

  case 52: /* power: TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN  */
#line 382 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PROHIBITION)) {
              yyerror("Type error in power expression");
          } else {
              (yyval.node) = create_legal(AST_PWR, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_POWER;
          }
      }
#line 1781 "parser.tab.c"
    break;

  case 53: /* power: TOKEN_PWR_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN  */
#line 390 "parser.y"
                                                                                               { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PRIVILEGE)) {
              yyerror("Type error in power expression");
          } else {
              (yyval.node) = create_legal(AST_PWR, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_POWER;
          }
      }
#line 1794 "parser.tab.c"
    break;

  case 54: /* liability: TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN  */
#line 401 "parser.y"
                                                                                                  { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in liability expression");
          } else {
              (yyval.node) = create_legal(AST_LIAB, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_LIABILITY;
          }
      }
#line 1807 "parser.tab.c"
    break;

  case 55: /* liability: TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN  */
#line 409 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_OBLIGATION)) {
              yyerror("Type error in liability expression");
          } else {
              (yyval.node) = create_legal(AST_LIAB, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_LIABILITY;
          }
      }
#line 1820 "parser.tab.c"
    break;

  case 56: /* liability: TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN  */
#line 417 "parser.y"
                                                                                                  { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PROHIBITION)) {
              yyerror("Type error in liability expression");
          } else {
              (yyval.node) = create_legal(AST_LIAB, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_LIABILITY;
          }
      }
#line 1833 "parser.tab.c"
    break;

  case 57: /* liability: TOKEN_LIAB_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN  */
#line 425 "parser.y"
                                                                                                { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PRIVILEGE)) {
              yyerror("Type error in liability expression");
          } else {
              (yyval.node) = create_legal(AST_LIAB, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_LIABILITY;
          }
      }
#line 1846 "parser.tab.c"
    break;

  case 58: /* disability: TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN  */
#line 436 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in disability expression");
          } else {
              (yyval.node) = create_legal(AST_DIS, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_DISABILITY;
          }
      }
#line 1859 "parser.tab.c"
    break;

  case 59: /* disability: TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN  */
#line 444 "parser.y"
                                                                                                { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_OBLIGATION)) {
              yyerror("Type error in disability expression");
          } else {
              (yyval.node) = create_legal(AST_DIS, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_DISABILITY;
          }
      }
#line 1872 "parser.tab.c"
    break;

  case 60: /* disability: TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN  */
#line 452 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PROHIBITION)) {
              yyerror("Type error in disability expression");
          } else {
              (yyval.node) = create_legal(AST_DIS, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_DISABILITY;
          }
      }
#line 1885 "parser.tab.c"
    break;

  case 61: /* disability: TOKEN_DIS_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN  */
#line 460 "parser.y"
                                                                                               { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PRIVILEGE)) {
              yyerror("Type error in disability expression");
          } else {
              (yyval.node) = create_legal(AST_DIS, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_DISABILITY;
          }
      }
#line 1898 "parser.tab.c"
    break;

  case 62: /* immunity: TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA claim_right TOKEN_CLOSE_PAREN  */
#line 471 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_CLAIM_RIGHT)) {
              yyerror("Type error in immunity expression");
          } else {
              (yyval.node) = create_legal(AST_IMM, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_IMMUNITY;
          }
      }
#line 1911 "parser.tab.c"
    break;

  case 63: /* immunity: TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA obligation TOKEN_CLOSE_PAREN  */
#line 479 "parser.y"
                                                                                                { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_OBLIGATION)) {
              yyerror("Type error in immunity expression");
          } else {
              (yyval.node) = create_legal(AST_IMM, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_IMMUNITY;
          }
      }
#line 1924 "parser.tab.c"
    break;

  case 64: /* immunity: TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA prohibition TOKEN_CLOSE_PAREN  */
#line 487 "parser.y"
                                                                                                 { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PROHIBITION)) {
              yyerror("Type error in immunity expression");
          } else {
              (yyval.node) = create_legal(AST_IMM, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_IMMUNITY;
          }
      }
#line 1937 "parser.tab.c"
    break;

  case 65: /* immunity: TOKEN_IMM_OPEN identifier TOKEN_COMMA identifier TOKEN_COMMA privilege TOKEN_CLOSE_PAREN  */
#line 495 "parser.y"
                                                                                               { 
          if (!check_expression_types((yyvsp[-5].node), TYPE_ASSET) || !check_expression_types((yyvsp[-3].node), TYPE_ASSET) || !check_expression_types((yyvsp[-1].node), TYPE_PRIVILEGE)) {
              yyerror("Type error in immunity expression");
          } else {
              (yyval.node) = create_legal(AST_IMM, create_binary_op(AST_COMMA, (yyvsp[-5].node), create_binary_op(AST_COMMA, (yyvsp[-3].node), (yyvsp[-1].node)))); 
              (yyval.node)->var_type = TYPE_IMMUNITY;
          }
      }
#line 1950 "parser.tab.c"
    break;

  case 66: /* query_expression: power  */
#line 505 "parser.y"
            { 
          if (!check_expression_types((yyvsp[0].node), TYPE_POWER)) {
              yyerror("Type mismatch in query expression: expected TYPE_POWER");
              YYERROR;
          } else {

              (yyval.node)->var_type = TYPE_QUERY; 
              (yyval.node) = create_query((yyvsp[0].node));
          }
      }
#line 1965 "parser.tab.c"
    break;

  case 67: /* query_expression: liability  */
#line 515 "parser.y"
                { 
          if (!check_expression_types((yyvsp[0].node), TYPE_LIABILITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_LIABILITY");
              YYERROR;
          } else {

              (yyval.node)->var_type = TYPE_QUERY; 
              (yyval.node) = create_query((yyvsp[0].node));              
          }
      }
#line 1980 "parser.tab.c"
    break;

  case 68: /* query_expression: disability  */
#line 525 "parser.y"
                 { 
          if (!check_expression_types((yyvsp[0].node), TYPE_DISABILITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_DISABILITY");
              YYERROR;
          } else {
              (yyval.node) = create_query((yyvsp[0].node));
              (yyval.node)->var_type = TYPE_QUERY; 
          }
      }
#line 1994 "parser.tab.c"
    break;

  case 69: /* query_expression: immunity  */
#line 534 "parser.y"
               { 
          if (!check_expression_types((yyvsp[0].node), TYPE_IMMUNITY)) {
              yyerror("Type mismatch in query expression: expected TYPE_IMMUNITY");
              YYERROR;
          } else {
              (yyval.node) = create_query((yyvsp[0].node));
              (yyval.node)->var_type = TYPE_QUERY; 
          }
      }
#line 2008 "parser.tab.c"
    break;

  case 70: /* asset_type: TOKEN_SERVICE  */
#line 546 "parser.y"
                    { (yyval.node) = create_type_node("Service"); }
#line 2014 "parser.tab.c"
    break;

  case 71: /* asset_type: TOKEN_PROPERTY  */
#line 547 "parser.y"
                     { (yyval.node) = create_type_node("Property"); }
#line 2020 "parser.tab.c"
    break;

  case 72: /* asset_subtype: TOKEN_NM  */
#line 551 "parser.y"
               { (yyval.node) = create_type_node("NM"); }
#line 2026 "parser.tab.c"
    break;

  case 73: /* asset_subtype: TOKEN_M  */
#line 552 "parser.y"
              { (yyval.node) = create_type_node("M"); }
#line 2032 "parser.tab.c"
    break;

  case 74: /* asset_subtype: TOKEN_PLUS  */
#line 553 "parser.y"
                 { (yyval.node) = create_type_node("+"); }
#line 2038 "parser.tab.c"
    break;

  case 75: /* asset_subtype: TOKEN_MINUS  */
#line 554 "parser.y"
                  { (yyval.node) = create_type_node("-"); }
#line 2044 "parser.tab.c"
    break;


#line 2048 "parser.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 557 "parser.y"



