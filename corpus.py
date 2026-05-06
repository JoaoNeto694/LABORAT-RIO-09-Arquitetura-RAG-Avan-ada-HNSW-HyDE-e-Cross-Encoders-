"""
Corpus simulado de fragmentos de manuais médicos.
Foco em jargão técnico (HyDE precisa traduzir linguagem coloquial para este vocabulário).
"""

CORPUS = [
    # Neurologia / Cefaleias
    "Enxaqueca sem aura é caracterizada por cefaleia pulsátil unilateral de intensidade moderada a severa, "
    "frequentemente acompanhada de fotofobia, fonofobia, náuseas e vômitos. A duração típica varia entre 4 e 72 horas "
    "quando não tratada. O tratamento abortivo de primeira linha inclui triptanos (sumatriptano 50-100mg VO) e AINEs.",

    "Cefaleia tensional episódica apresenta dor bilateral em aperto ou pressão, de intensidade leve a moderada, "
    "sem náuseas significativas. Não é agravada por atividade física rotineira. O manejo profilático envolve "
    "amitriptilina em baixas doses e técnicas de relaxamento muscular cervical.",

    "Cefaleia em salvas (cluster) manifesta-se com dor periorbitária excruciante unilateral, associada a "
    "sintomas autonômicos ipsilaterais como lacrimejamento, congestão nasal, ptose e miose (síndrome de Horner parcial). "
    "Crises duram 15 a 180 minutos e respondem a oxigenoterapia 100% a 12L/min.",

    # Cardiologia
    "Síndrome coronariana aguda (SCA) com supradesnivelamento do segmento ST exige reperfusão coronariana imediata, "
    "preferencialmente por angioplastia primária em até 90 minutos do primeiro contato médico. Marcadores de necrose "
    "miocárdica como troponina I ultrassensível confirmam o infarto agudo do miocárdio.",

    "Insuficiência cardíaca com fração de ejeção reduzida (ICFEr) é definida por FEVE inferior a 40% ao ecocardiograma. "
    "A terapia otimizada inclui IECA ou BRA, betabloqueadores, antagonistas mineralocorticoides e iSGLT2. "
    "Sintomas incluem dispneia aos esforços, ortopneia e edema de membros inferiores.",

    "Fibrilação atrial paroxística é a arritmia supraventricular sustentada mais prevalente, caracterizada por "
    "atividade elétrica atrial caótica. A estratificação de risco tromboembólico utiliza o escore CHA2DS2-VASc, "
    "indicando anticoagulação oral em pacientes com pontuação ≥ 2.",

    # Pneumologia
    "Doença pulmonar obstrutiva crônica (DPOC) caracteriza-se por limitação crônica e progressiva ao fluxo aéreo, "
    "diagnosticada por espirometria com VEF1/CVF pós-broncodilatador < 0,70. Exacerbações agudas demandam "
    "broncodilatadores de curta ação, corticosteroides sistêmicos e antibioticoterapia quando indicado.",

    "Asma brônquica é doença inflamatória crônica das vias aéreas com hiper-responsividade brônquica reversível. "
    "Sintomas incluem sibilância, dispneia, opressão torácica e tosse, frequentemente noturna. O controle utiliza "
    "corticosteroides inalatórios associados a beta-2 agonistas de longa ação (LABA).",

    "Tromboembolismo pulmonar (TEP) deve ser suspeitado em pacientes com dispneia súbita, dor torácica pleurítica "
    "e hemoptise. O escore de Wells estratifica probabilidade pré-teste, e o D-dímero possui alto valor preditivo "
    "negativo. Angiotomografia de tórax confirma o diagnóstico.",

    # Gastroenterologia
    "Doença do refluxo gastroesofágico (DRGE) manifesta-se classicamente por pirose retroesternal e regurgitação ácida. "
    "Sintomas atípicos incluem tosse crônica, rouquidão e dor torácica não cardíaca. O tratamento de primeira linha "
    "consiste em inibidores da bomba de prótons (IBP) por 8 semanas.",

    "Síndrome do intestino irritável (SII) é distúrbio funcional caracterizado por dor abdominal recorrente associada "
    "à defecação, com alteração na frequência ou forma das fezes. Os critérios de Roma IV orientam o diagnóstico. "
    "O manejo envolve dieta com baixo teor de FODMAPs e antiespasmódicos como brometo de otilônio.",

    # Endocrinologia
    "Diabetes mellitus tipo 2 é caracterizado por resistência insulínica periférica e disfunção progressiva das células beta "
    "pancreáticas. Diagnóstico estabelecido por glicemia de jejum ≥ 126 mg/dL ou HbA1c ≥ 6,5%. "
    "Metformina permanece como terapia farmacológica de primeira linha.",

    "Hipotireoidismo primário decorre de falência da glândula tireoide, mais comumente por tireoidite de Hashimoto. "
    "Manifesta-se com fadiga, intolerância ao frio, ganho ponderal, bradicardia e mixedema. "
    "Diagnóstico laboratorial: TSH elevado com T4 livre reduzido. Reposição com levotiroxina sódica.",

    # Infectologia
    "Pneumonia adquirida na comunidade (PAC) em adultos imunocompetentes tem como agentes etiológicos mais frequentes "
    "Streptococcus pneumoniae, Mycoplasma pneumoniae e Haemophilus influenzae. O escore CURB-65 orienta a decisão "
    "entre tratamento ambulatorial e hospitalar. Amoxicilina é primeira escolha em casos leves.",

    "Infecção do trato urinário (ITU) baixa não complicada em mulheres apresenta disúria, polaciúria e urgência miccional. "
    "Escherichia coli é o patógeno predominante. Tratamento empírico com nitrofurantoína 100mg 6/6h por 5 dias "
    "ou fosfomicina trometamol 3g em dose única.",

    # Reumatologia
    "Artrite reumatoide é doença autoimune sistêmica com poliartrite simétrica de pequenas articulações, rigidez matinal "
    "superior a uma hora e fator reumatoide ou anti-CCP positivos. O tratamento precoce com DMARDs sintéticos como "
    "metotrexato previne erosões articulares e incapacidade funcional.",

    # Nefrologia
    "Doença renal crônica (DRC) é definida por taxa de filtração glomerular estimada (TFGe) inferior a 60 mL/min/1,73m² "
    "por mais de três meses ou marcadores de lesão renal persistentes. O estadiamento KDIGO orienta o manejo. "
    "Albuminúria é marcador prognóstico independente de progressão e mortalidade cardiovascular.",

    # Psiquiatria
    "Transtorno depressivo maior caracteriza-se por humor deprimido e/ou anedonia por período mínimo de duas semanas, "
    "associado a sintomas neurovegetativos como alteração do sono, apetite, energia e concentração. "
    "Inibidores seletivos da recaptação de serotonina (ISRS) como sertralina são primeira escolha farmacológica.",

    "Transtorno de ansiedade generalizada (TAG) apresenta preocupação excessiva e persistente por mais de seis meses, "
    "com sintomas somáticos como tensão muscular, irritabilidade, fadiga e distúrbios do sono. "
    "Terapia cognitivo-comportamental e ISRS/IRSN compõem o tratamento de primeira linha.",

    # Dermatologia
    "Dermatite atópica é doença inflamatória cutânea crônica e pruriginosa, com xerose e lesões eczematosas em "
    "distribuição flexural característica. Associa-se à marcha atópica, com risco aumentado de asma e rinite alérgica. "
    "Hidratação intensiva e corticosteroides tópicos de baixa potência compõem o tratamento basal.",
]
