"""
Xplore DS :: Script Template
"""

# importando as bibliotecas padrao
import sys, os
from pathlib import Path
from dotenv import load_dotenv


# Configurando path para raiz do projeto e setup de reconhecimento da pasta da lib
project_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(project_folder))


from xplore_ds.environment.environment import XploreDSLocalhost
from xplore_ds.environment.logging import XploreDSLogging


# ==================================================================================
# Setup do script

script_name = os.path.basename(__file__)


# Variaveis de ambiente
load_dotenv()

# Criando estrutura de execucao local
env = XploreDSLocalhost(run_folder=project_folder)

# Criando estrutura de logs
log = XploreDSLogging(project_root=project_folder, script_name=script_name)
log.init_run()

# ==================================================================================

# ==================================================================================
# Funcoes do script
# ==================================================================================

# ==================================================================================
# Parametrizacao do script
# ==================================================================================

output_folder = "output"

# ==================================================================================
# Regras de negócio
# ==================================================================================


# ==================================================================================
# Encerramento do script
# ==================================================================================
log.close_run()
