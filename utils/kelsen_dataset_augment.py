import json
from tqdm import tqdm
import random
import argparse
thesaurus = {
    # Actors
    'comprador': {'gender': 'm', 'synonyms': ['adquirente', 'comprador', 'parte compradora', 'adquisidor', 'cliente', 'demandante']},
    'vendedor': {'gender': 'm', 'synonyms': ['enajenante', 'vendedor', 'parte vendedora', 'transmitente', 'proveedor', 'ofertante']},
    'acreedor': {'gender': 'm', 'synonyms': ['titular del crédito', 'parte acreedora', 'beneficiario', 'creditor', 'prestamista']},
    'deudor': {'gender': 'm', 'synonyms': ['obligado', 'parte deudora', 'sujeto pasivo de la obligación', 'deudor', 'pagador']},
    'contratante': {'gender': 'm', 'synonyms': ['parte contratante', 'otorgante', 'interviniente', 'signatario', 'firmante']},
    'tercero': {'gender': 'm', 'synonyms': ['tercera parte', 'tercera persona', 'ajeno al contrato', 'extracontractual', 'independiente']},
    'beneficiario': {'gender': 'm', 'synonyms': ['favorecido', 'receptor', 'destinatario', 'perceptor', 'agraciado']},
    'fiador': {'gender': 'm', 'synonyms': ['garante', 'aval', 'segurador', 'codeudor solidario', 'fiador']},
    'asegurado': {'gender': 'm', 'synonyms': ['titular de la póliza', 'parte asegurada', 'protegido', 'beneficiario de la póliza', 'asegurado']},
    'arrendador': {'gender': 'm', 'synonyms': ['propietario', 'dueño', 'arrendador', 'locador', 'cedente']},
    'arrendatario': {'gender': 'm', 'synonyms': ['inquilino', 'arrendatario', 'locatario', 'usuario']},

    # Legal concepts
    'contrato': {'gender': 'm', 'synonyms': ['acuerdo', 'convenio', 'pacto', 'negocio jurídico', 'entendimiento']},
    'cosa': {'gender': 'f', 'synonyms': ['objeto', 'bien', 'artículo', 'mercancía', 'elemento']},
    'precio': {'gender': 'm', 'synonyms': ['costo', 'importe', 'valor', 'contraprestación', 'tarifa', 'suma']},
    'pago': {'gender': 'm', 'synonyms': ['desembolso', 'abono', 'liquidación', 'satisfacción de la deuda', 'cancelación']},
    'obligación': {'gender': 'f', 'synonyms': ['compromiso', 'deber', 'responsabilidad', 'carga', 'mandato', 'deber jurídico']},
    'derecho': {'gender': 'm', 'synonyms': ['facultad', 'prerrogativa', 'atribución', 'potestad', 'derecho subjetivo', 'competencia']},
    'propiedad': {'gender': 'f', 'synonyms': ['dominio', 'posesión', 'titularidad', 'pertenencia', 'bien inmueble', 'propiedad privada']},
    'entrega': {'gender': 'f', 'synonyms': ['traspaso', 'transferencia', 'cesión', 'transmisión', 'consignación', 'tradición']},
    'incumplimiento': {'gender': 'm', 'synonyms': ['inejecución', 'inobservancia', 'contravención', 'violación', 'incumplimiento contractual']},
    'rescisión': {'gender': 'f', 'synonyms': ['resolución', 'terminación', 'disolución del contrato', 'anulación', 'extinción']},
    'vicio': {'gender': 'm', 'synonyms': ['defecto', 'imperfección', 'falla', 'anomalía', 'taras', 'vicio oculto']},
    'evicción': {'gender': 'f', 'synonyms': ['desposesión legal', 'pérdida del derecho', 'privación de la cosa', 'reclamación judicial']},
    'mora': {'gender': 'f', 'synonyms': ['retraso', 'demora', 'dilación en el cumplimiento', 'morosidad']},
    'daños y perjuicios': {'gender': 'm', 'synonyms': ['indemnización', 'resarcimiento', 'compensación por daños', 'reparación']},
    'buena fe': {'gender': 'f', 'synonyms': ['lealtad', 'probidad', 'rectitud en la conducta', 'honradez']},
    'mala fe': {'gender': 'f', 'synonyms': ['dolo', 'mala intención', 'conducta fraudulenta', 'engaño']},
    'litigio': {'gender': 'm', 'synonyms': ['controversia', 'disputa legal', 'pleito', 'conflicto judicial']},
    'cláusula': {'gender': 'f', 'synonyms': ['estipulación', 'disposición contractual', 'pacto específico', 'condición']},
    'notificación': {'gender': 'f', 'synonyms': ['comunicación', 'aviso', 'intimación', 'información formal']},
    'prescripción': {'gender': 'f', 'synonyms': ['caducidad', 'extinción del derecho', 'pérdida del derecho por el transcurso del tiempo']},
    'jurisdicción': {'gender': 'f', 'synonyms': ['competencia', 'ámbito judicial', 'territorio jurisdiccional', 'autoridad judicial']},
    'traslación de la propiedad': {'gender': 'f', 'synonyms': ['transferencia de la propiedad', 'transmisión del dominio', 'cesión de derechos de propiedad']},
    'registro público': {'gender': 'm', 'synonyms': ['registro de la propiedad', 'registro público de la propiedad', 'registro civil', 'registro oficial']},
    'tradición': {'gender': 'f', 'synonyms': ['entrega física', 'entrega simbólica', 'transmisión de la posesión', 'acto de entrega']},
    'disposición': {'gender': 'f', 'synonyms': ['norma', 'regla', 'regulación', 'precepto']},
    
    # Actions and verbs (no gender)
    'debe': {'gender': None, 'synonyms': ['está obligado a', 'tiene que', 'ha de', 'le corresponde', 'deberá']},
    'vender': {'gender': None, 'synonyms': ['enajenar', 'transmitir', 'ceder', 'traspasar', 'comercializar']},
    'comprar': {'gender': None, 'synonyms': ['adquirir', 'obtener', 'hacerse con', 'procurarse', 'comprar']},
    'pagar': {'gender': None, 'synonyms': ['abonar', 'satisfacer', 'liquidar', 'desembolsar', 'cancelar']},
    'entregar': {'gender': None, 'synonyms': ['transferir', 'ceder', 'traspasar', 'transmitir', 'consignar']},
    'recibir': {'gender': None, 'synonyms': ['aceptar', 'tomar', 'obtener', 'adquirir', 'recoger']},
    'incumplir': {'gender': None, 'synonyms': ['contravenir', 'infringir', 'violar', 'quebrantar', 'no cumplir']},
    'rescindir': {'gender': None, 'synonyms': ['resolver', 'terminar', 'disolver', 'anular', 'revocar']},
    'estipular': {'gender': None, 'synonyms': ['acordar', 'pactar', 'convenir', 'establecer', 'prescribir']},
    'demandar': {'gender': None, 'synonyms': ['reclamar', 'exigir', 'presentar una demanda', 'accionar judicialmente']},
    'garantizar': {'gender': None, 'synonyms': ['asegurar', 'proteger', 'respaldar', 'afianzar']},
    'notificar': {'gender': None, 'synonyms': ['comunicar', 'informar', 'avisar', 'dar aviso', 'notificar']},
    'registrar': {'gender': None, 'synonyms': ['inscribir', 'anotar', 'documentar', 'certificar']},
    'prescribir': {'gender': None, 'synonyms': ['caducar', 'extinguirse', 'expirar', 'precluir']},
    'litigar': {'gender': None, 'synonyms': ['demandar', 'pleitear', 'disputar', 'controvertir']},
    'transferir': {'gender': None, 'synonyms': ['ceder', 'traspasar', 'transmitir', 'desplazar']},
    'traditar': {'gender': None, 'synonyms': ['entregar', 'transferir la posesión', 'realizar la tradición']},
    'formalizar': {'gender': None, 'synonyms': ['legalizar', 'autenticar', 'validar', 'ratificar']},
    'ejecutar': {'gender': None, 'synonyms': ['realizar', 'cumplir', 'llevar a cabo', 'implementar']},
    'indemnizar': {'gender': None, 'synonyms': ['resarcir', 'compensar', 'reparar', 'pagar daños y perjuicios']},
    'conservar': {'gender': None, 'synonyms': ['mantener', 'preservar', 'guardar', 'retener']},
    'revocar': {'gender': None, 'synonyms': ['anular', 'invalidar', 'cancelar', 'derogar']},
    'aprobar': {'gender': None, 'synonyms': ['ratificar', 'sancionar', 'confirmar', 'consentir']},
    'restituir': {'gender': None, 'synonyms': ['devolver', 'reintegrar', 'restaurar', 'resarcir']},
    'ratificar': {'gender': None, 'synonyms': ['confirmar', 'reafirmar', 'aprobar', 'sancionar']},
    'anular': {'gender': None, 'synonyms': ['invalidar', 'revocar', 'cancelar', 'rescindir']},
    'autorizar': {'gender': None, 'synonyms': ['permitir', 'consentir', 'dar permiso', 'facultar']},
    'delegar': {'gender': None, 'synonyms': ['transferir', 'ceder', 'asignar', 'encargar']},
    'impugnar': {'gender': None, 'synonyms': ['contestar', 'refutar', 'oponerse', 'desafiar']},
    'exigir': {'gender': None, 'synonyms': ['demandar', 'requerir', 'solicitar', 'pedir']},
    'contratar': {'gender': None, 'synonyms': ['firmar un contrato', 'pactar', 'convenir', 'acordar']},
    'cumplir': {'gender': None, 'synonyms': ['ejecutar', 'realizar', 'llevar a cabo', 'obedecer']},
    'infringir': {'gender': None, 'synonyms': ['violar', 'contravenir', 'transgredir', 'quebrantar']},
    'resolver': {'gender': None, 'synonyms': ['solucionar', 'disponer', 'rescindir', 'decidir']},
    'transmitir': {'gender': None, 'synonyms': ['transferir', 'ceder', 'pasar', 'comunicar']},
    'cesar': {'gender': None, 'synonyms': ['terminar', 'detener', 'acabar', 'interrumpir']},
    'sancionar': {'gender': None, 'synonyms': ['castigar', 'penalizar', 'aprobar', 'ratificar']},
    'implementar': {'gender': None, 'synonyms': ['ejecutar', 'aplicar', 'realizar', 'llevar a cabo']},
    'liquidar': {'gender': None, 'synonyms': ['pagar', 'abonar', 'saldar', 'ajustar cuentas']},
    'convenir': {'gender': None, 'synonyms': ['acordar', 'pactar', 'establecer', 'aceptar']},
    'extinguir': {'gender': None, 'synonyms': ['finalizar', 'terminar', 'cancelar', 'anular']},
    'disolver': {'gender': None, 'synonyms': ['terminar', 'rescindir', 'anular', 'deshacer']},
     'debe': {
        'gender': None, 
        'synonyms': ['está obligado a', 'tiene que', 'ha de', 'le corresponde']
    },
    'debiendo': {
        'gender': None,
        'synonyms': ['estando obligado a', 'teniendo que', 'habiendo de', 'correspondiendo']
    },
    'debido': {
        'gender': 'm', 
        'synonyms': ['obligado', 'ordenado', 'prescrito', 'exigido']
    },
    'debida': {
        'gender': 'f', 
        'synonyms': ['obligada', 'ordenada', 'prescrita', 'exigida']
    },
    'vender': {
        'gender': None,
        'synonyms': ['enajenar', 'transmitir', 'ceder', 'traspasar']
    },
    'vendiendo': {
        'gender': None,
        'synonyms': ['enajenando', 'transmitiendo', 'cediendo', 'traspasando']
    },
    'vendido': {
        'gender': 'm', 
        'synonyms': ['enajenado', 'transmitido', 'cedido', 'traspasado']
    },
    'vendida': {
        'gender': 'f', 
        'synonyms': ['enajenada', 'transmitida', 'cedida', 'traspasada']
    },
    'comprar': {
        'gender': None,
        'synonyms': ['adquirir', 'obtener', 'hacerse con', 'procurarse']
    },
    'comprando': {
        'gender': None,
        'synonyms': ['adquiriendo', 'obteniendo', 'haciéndose con', 'procurándose']
    },
    'comprado': {
        'gender': 'm', 
        'synonyms': ['adquirido', 'obtenido', 'conseguido', 'procurado']
    },
    'comprada': {
        'gender': 'f', 
        'synonyms': ['adquirida', 'obtenida', 'conseguida', 'procurada']
    },
    'pagar': {
        'gender': None,
        'synonyms': ['abonar', 'satisfacer', 'liquidar', 'desembolsar']
    },
    'pagando': {
        'gender': None,
        'synonyms': ['abonando', 'satisfaciendo', 'liquidando', 'desembolsando']
    },
    'pagado': {
        'gender': 'm', 
        'synonyms': ['abonado', 'satisfecho', 'liquidado', 'desembolsado']
    },
    'pagada': {
        'gender': 'f', 
        'synonyms': ['abonada', 'satisfecha', 'liquidada', 'desembolsada']
    },
    'entregar': {
        'gender': None,
        'synonyms': ['transferir', 'ceder', 'traspasar', 'transmitir', 'consignar', 'traditar']
    },
    'entregando': {
        'gender': None,
        'synonyms': ['transfiriendo', 'cediendo', 'traspasando', 'transmitiendo', 'consignando', 'traditando']
    },
    'entregado': {
        'gender': 'm', 
        'synonyms': ['transferido', 'cedido', 'traspasado', 'transmitido', 'consignado', 'traditado']
    },
    'entregada': {
        'gender': 'f', 
        'synonyms': ['transferida', 'cedida', 'traspasada', 'transmitida', 'consignada', 'traditada']
    },
    'recibir': {
        'gender': None,
        'synonyms': ['aceptar', 'tomar', 'obtener', 'adquirir']
    },
    'recibiendo': {
        'gender': None,
        'synonyms': ['aceptando', 'tomando', 'obteniendo', 'adquiriendo']
    },
    'recibido': {
        'gender': 'm', 
        'synonyms': ['aceptado', 'tomado', 'obtenido', 'adquirido']
    },
    'recibida': {
        'gender': 'f', 
        'synonyms': ['aceptada', 'tomada', 'obtenida', 'adquirida']
    },
    'incumplir': {
        'gender': None,
        'synonyms': ['contravenir', 'infringir', 'violar', 'quebrantar']
    },
    'incumpliendo': {
        'gender': None,
        'synonyms': ['contraviniendo', 'infringiendo', 'violando', 'quebrantando']
    },
    'incumplido': {
        'gender': 'm', 
        'synonyms': ['contravenido', 'infringido', 'violado', 'quebrantado']
    },
    'incumplida': {
        'gender': 'f', 
        'synonyms': ['contravenida', 'infringida', 'violada', 'quebrantada']
    },
    'rescindir': {
        'gender': None,
        'synonyms': ['resolver', 'terminar', 'disolver', 'anular']
    },
    'rescindiendo': {
        'gender': None,
        'synonyms': ['resolviendo', 'terminando', 'disolviendo', 'anulando']
    },
    'rescindido': {
        'gender': 'm', 
        'synonyms': ['resuelto', 'terminado', 'disuelto', 'anulado']
    },
    'rescindida': {
        'gender': 'f', 
        'synonyms': ['resuelta', 'terminada', 'disuelta', 'anulada']
    },
    'estipular': {
        'gender': None,
        'synonyms': ['acordar', 'pactar', 'convenir', 'establecer']
    },
    'estipulando': {
        'gender': None,
        'synonyms': ['acordando', 'pactando', 'conviniendo', 'estableciendo']
    },
    'estipulado': {
        'gender': 'm', 
        'synonyms': ['acordado', 'pactado', 'convenido', 'establecido']
    },
    'estipulada': {
        'gender': 'f', 
        'synonyms': ['acordada', 'pactada', 'convenida', 'establecida']
    },
    'obligado': {
        'gender': 'm', 
        'synonyms': ['consternido', 'mandado']
    },
    'obligada': {
        'gender': 'f', 
        'synonyms': ['consternida', 'mandada']
        }

}

multi_word_phrases = {
    'ser obligado': [
        'estar consternido', 
        'estar mandado', 
        'tener la obligación de', 
        'verse forzado a',
        'estar compelido a',
        'ser coaccionado a',
        'tener que cumplir con',
    ],
    'estar obligado a': [
        'tener que', 
        'deber', 
        'ser necesario que', 
        'verse en la necesidad de',
        'tener la obligación de',
        'estar en la obligación de',
        'tener que realizar',
    ],
    'tener que': [
        'estar obligado a', 
        'deber', 
        'ser necesario que', 
        'verse en la necesidad de',
        'tener que cumplir con',
        'tener que llevar a cabo',
        'ser preciso que',
    ],
    'ha de': [
        'debe', 
        'tiene la obligación de', 
        'es necesario que', 
        'se ve en la necesidad de',
        'debería',
        'es imperativo que',
        'es obligado que',
    ],
    'puede ser': [
        'es posible que sea', 
        'tiene la posibilidad de ser', 
        'es factible que sea',
        'podría ser',
        'existe la posibilidad de que sea',
        'es viable que',
    ],
    'no puede': [
        'está impedido de', 
        'tiene prohibido', 
        'no está autorizado a',
        'se le prohíbe',
        'está restringido a',
        'no está en posición de',
    ],
    'debe pagar': [
        'está obligado a abonar', 
        'tiene que liquidar', 
        'ha de satisfacer',
        'está en la obligación de pagar',
        'debe cancelar',
        'tiene el deber de pagar',
    ],
    'puede vender': [
        'tiene la facultad de enajenar', 
        'está autorizado a transferir', 
        'tiene derecho a ceder',
        'puede comercializar',
        'tiene la posibilidad de vender',
        'está en condiciones de vender',
    ],
    'debe entregar': [
        'está obligado a traspasar', 
        'tiene que transferir', 
        'ha de ceder',
        'debe proporcionar',
        'está en la obligación de entregar',
        'tiene que remitir',
    ],
    'puede recibir': [
        'está facultado para aceptar', 
        'tiene derecho a tomar', 
        'está autorizado a obtener',
        'puede aceptar',
        'tiene la posibilidad de recibir',
        'está en condiciones de recibir',
    ],
    'debe cumplir': [
        'está obligado a ejecutar', 
        'tiene que llevar a cabo', 
        'ha de realizar',
        'debe efectuar',
        'está en la obligación de cumplir',
        'tiene la obligación de cumplir',
    ],
    'puede rescindir': [
        'tiene la facultad de resolver', 
        'está autorizado a terminar', 
        'tiene derecho a anular',
        'puede cancelar',
        'tiene la posibilidad de rescindir',
        'está en posición de rescindir',
    ],
    # New phrases based on the dataset
    'cosa cierta': [
        'bien determinado',
        'objeto específico',
        'artículo concreto',
        'elemento definido',
        'mercancía particular',
    ],
    'traslación de dominio': [
        'transferencia de propiedad',
        'cesión de derechos reales',
        'transmisión de titularidad',
        'enajenación de posesión',
    ],
    'no acepte': [
        'rechace',
        'no admita',
        'se niegue a recibir',
        'decline',
    ],
    'de mayor valor': [
        'más valioso',
        'de superior precio',
        'de mayor costo',
        'de más alto importe',
    ],
    'entregar sus accesorios': [
        'proporcionar sus complementos',
        'ceder sus elementos adicionales',
        'transferir sus componentes asociados',
        'dar sus partes complementarias',
    ],
    'salvo que': [
        'excepto si',
        'a menos que',
        'con la excepción de que',
        'exceptuando que',
    ],
    'título de la obligación': [
        'documento del compromiso',
        'instrumento de la responsabilidad',
        'escrito del deber',
        'acta de la obligación',
    ],
    'circunstancias del caso': [
        'condiciones de la situación',
        'particularidades del asunto',
        'detalles del contexto',
        'características específicas',
    ],
    'enajenaciones de cosas': [
        'ventas de bienes',
        'transferencias de objetos',
        'cesiones de artículos',
        'transmisiones de propiedades',
    ],
    'mero efecto del contrato': [
        'simple consecuencia del acuerdo',
        'puro resultado del convenio',
        'sola derivación del pacto',
        'mera implicación del trato',
    ],
    'sin dependencia de tradición': [
        'independientemente de la entrega',
        'sin necesidad de traspaso físico',
        'al margen de la transferencia material',
        'prescindiendo de la transmisión efectiva',
    ],
    'Registro Público': [
        'Registro de la Propiedad',
        'Registro Oficial',
        'Registro Estatal',
        'Registro Gubernamental',
    ],
    'especie indeterminada': [
        'tipo no específico',
        'clase indefinida',
        'categoría no determinada',
        'género impreciso',
    ],
    'con conocimiento del acreedor': [
        'bajo el entendimiento del titular del crédito',
        'con la cognición del beneficiario',
        'estando al tanto la parte acreedora',
        'con la información del prestamista',
    ],
    'mediana calidad': [
        'calidad promedio',
        'calidad estándar',
        'calidad intermedia',
        'calidad regular',
    ],
    'por culpa del deudor': [
        'debido a la negligencia del obligado',
        'por responsabilidad de la parte deudora',
        'a causa del incumplimiento del deudor',
        'por falta atribuible al sujeto pasivo',
    ],
    'daños y perjuicios': [
        'indemnización compensatoria',
        'reparación de daños',
        'compensación por pérdidas',
        'resarcimiento económico',
    ],
    'caso fortuito o fuerza mayor': [
        'evento imprevisto o inevitable',
        'circunstancia extraordinaria e imprevisible',
        'suceso fuera del control de las partes',
        'acontecimiento imprevisible e irresistible',
    ],
    'precio cierto': [
        'valor determinado',
        'costo específico',
        'importe definido',
        'monto concreto',
    ],
    'en dinero': [
        'en efectivo',
        'en moneda de curso legal',
        'en especie monetaria',
        'en divisa',
    ],
        'traslación de dominio': ['transferencia de propiedad', 'transmisión de dominio', 'cambio de titularidad'],
    'no aceptar cosa distinta': ['no recibir objeto diferente', 'rechazar cosa diferente', 'no admitir cosa diversa'],
    'entregar accesorios': ['proporcionar añadidos', 'incluir complementos', 'ceder accesorios'],
    'cumplir la condición': ['satisfacer la condición', 'llenar el requisito', 'realizar la condición'],
    'asumir la pérdida': ['aceptar la pérdida', 'hacerse cargo de la pérdida', 'tomar responsabilidad por la pérdida'],
    'exigir el pago': ['demandar el pago', 'solicitar el abono', 'reclamar el desembolso'],
    'revocar la oferta': ['anular la oferta', 'cancelar el ofrecimiento', 'retirar la propuesta'],
    'debe cumplir con': ['está obligado a cumplir', 'tiene la obligación de realizar', 'debe llevar a cabo'],
    'pagar intereses': ['abonar intereses', 'satisfacer los intereses', 'liquidar los intereses'],
    'suspender el pago': ['detener el pago', 'interrumpir el desembolso', 'pausar el abono'],
    'asegurar la posesión': ['garantizar la posesión', 'proteger la posesión', 'resguardar la posesión'],
    'formalizar el contrato': ['oficializar el acuerdo', 'legalizar el convenio', 'validar el contrato'],
    'otorgar documento privado': ['conceder documento privado', 'expedir documento privado', 'emitir documento privado'],
    'dividir la recompensa': ['repartir la recompensa', 'fraccionar la recompensa', 'segmentar la recompensa'],
    'designar un juez': ['nombrar un árbitro', 'asignar un juez', 'seleccionar un juez'],
    'oponer excepciones': ['interponer objeciones', 'presentar excepciones', 'formular excepciones'],
    'no declarar el litigio': ['no revelar el litigio', 'no informar sobre el litigio', 'ocultar el litigio'],
    'cumplir con la obligación': ['satisfacer la obligación', 'realizar la obligación', 'ejecutar la obligación'],
    'rescindir el contrato': ['anular el contrato', 'resolver el contrato', 'cancelar el contrato'],
    'puede exigir': ['tiene derecho a exigir', 'está autorizado a exigir', 'está facultado para exigir'],
    'no puede recobrar': ['no está en posición de recuperar', 'no puede recuperar', 'no está autorizado a recobrar'],
    'procede con dolo': ['actúa con mala fe', 'opera con dolo', 'se comporta con dolo'],
    'venda bienes': ['enajene bienes', 'transfiera bienes', 'ceda bienes']
}


def word_substitute(text):
    words = text.split()
    i = 0
    changes_made = False
    
    while i < len(words):
        # Check for multi-word phrases first
        for j in range(4, 0, -1):  # Check for phrases up to 4 words long
            if i + j <= len(words):
                phrase = ' '.join(words[i:i+j]).lower()
                if phrase in multi_word_phrases:
                    new_phrase = random.choice(multi_word_phrases[phrase])
                    words[i:i+j] = new_phrase.split()
                    changes_made = True
                    i += len(new_phrase.split())
                    break
        else:
            # If no multi-word phrase found, check individual words
            word = words[i]
            lower_word = word.lower()
            if lower_word in thesaurus:
                gender = thesaurus[lower_word]['gender']
                synonyms = [s for s in thesaurus[lower_word]['synonyms'] if s.lower() != lower_word]
                if gender is not None:
                    synonyms = [s for s in synonyms if thesaurus.get(s.lower(), {}).get('gender') == gender]
                if synonyms:
                    new_word = random.choice(synonyms)
                    words[i] = new_word if word.istitle() else new_word.lower()
                    changes_made = True
            i += 1
    
    return ' '.join(words), changes_made

def paraphrase(text):
    attempts = 0
    max_attempts = 100  # Prevent infinite loop
    while attempts < max_attempts:
        new_text, changes_made = word_substitute(text)
        if changes_made:
            return new_text
        attempts += 1
    return text  # Return original if no changes could be made after max attempts


def process_translation(translation, num_paraphrases=5, auto_approve=False):
    original_text = translation['original']
    results = [translation]  # Include the original translation
    
    approved_paraphrases = 0
    attempts = 0
    max_attempts = num_paraphrases * 2  # Allow more attempts than required paraphrases

    while approved_paraphrases < num_paraphrases and attempts < max_attempts:
        paraphrased_text = paraphrase(original_text)
        if paraphrased_text != original_text:  # Ensure the paraphrase is different from the original
            if auto_approve or get_user_approval(original_text, paraphrased_text):
                paraphrased_translation = translation.copy()
                paraphrased_translation['original'] = paraphrased_text
                paraphrased_translation['article_number'] += f"-{approved_paraphrases + 1}"
                results.append(paraphrased_translation)
                approved_paraphrases += 1
        attempts += 1

    return results
    
def get_user_approval(original, paraphrase):
    print(f"\nOriginal: {original}")
    print(f"Paraphrase: {paraphrase}")
    while True:
        response = input("Approve this paraphrase? (y/n): ").lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Please enter 'y' for yes or 'n' for no.")



def augment_dataset(input_file, output_file, num_paraphrases=5, auto_approve=False):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented_translations = []
    
    for translation in tqdm(data['translations'], desc="Processing translations"):
        augmented_translations.extend(process_translation(translation, num_paraphrases, auto_approve))
    
    augmented_data = {'translations': augmented_translations}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print(f"Augmented dataset saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Augment Kelsen dataset with paraphrases")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSON file")
    parser.add_argument("-n", "--num_paraphrases", type=int, default=5, help="Number of paraphrases to generate for each translation")
    parser.add_argument("-a", "--auto", action="store_true", help="Automatically approve all paraphrases without user interaction")
    
    args = parser.parse_args()
    
    augment_dataset(args.input_file, args.output_file, args.num_paraphrases, args.auto)

if __name__ == "__main__":
    main()
