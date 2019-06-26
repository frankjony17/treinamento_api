from flask import Flask, send_from_directory
from flask_restplus import Api, Resource, fields
from werkzeug.exceptions import NotFound

from resources.treinamento import Treinamento

DOWNLOAD_DIRECTORY = "model/"
PATH = "svm_model_anisotropic.pickle"

app = Flask(__name__)
api = Api(app, version='1.0', title='Treinamento API', description='Treinamento anisotropic svm API')
ns = api.namespace('svm_api', description='TODO operações')
app.config.RESTPLUS_MASK_SWAGGER = False

svm_api = api.model('SVM', {
    'endereco': fields.String(required=True, description='Endereço das imagens'),
    'start_learning': fields.String(required=False),
    'stop_learning': fields.String(required=False),
    'elapsed_learning': fields.String(required=False),
    # 'classification_report_classifier': fields.List,
    'accuracy': fields.String(required=False),
    'confusion_matrix': fields.String(required=False),
    'filename': fields.String(required=False)
})

parser = api.parser()
parser.add_argument('endereco', type=str, required=True, help='Endereço das imagens (não pode ficar em branco)', location='form')


def init_treinamento(model):
    tr = Treinamento(model)
    if tr.bool:
        tr.learning()
        tr.testing_model()
        ac = tr.save_model()
        model = ac.mapper(model)
    return {
        'bool': tr.bool,
        'model': model
    }


@ns.route('/')
@ns.response(200, 'Treinamento concluído com sucesso')
@ns.response(404, 'Nenhuma imagem encontrada no endereço especificado')
class Svm(Resource):
    @ns.doc('comecar_treinamento')
    @ns.doc(parser=parser)
    def post(self):
        """Começar a treinar"""
        code = 200
        args = parser.parse_args()
        resposta = init_treinamento(args)
        if resposta['bool'] is False:
            ns.abort(404, 'Nenhuma imagem encontrada no endereço especificado')
        return resposta['model'], code


@ns.route('/file')
@ns.response(200, 'Treinamento concluído com sucesso, retorna arquivo \'.pickle\' gerado')
@ns.response(404, 'Arquivo não gerado.')
class SvmFile(Resource):
    @ns.doc('comecar_treinamento_file')
    @ns.doc(parser=parser)
    def post(self):
        """Começar a treinar (retorna arquivo 'pickle' gerado)"""
        args = parser.parse_args()
        resposta = init_treinamento(args)
        if resposta['bool'] is False:
            ns.abort(404, 'Nenhuma imagem encontrada no endereço especificado')

        try:
            return send_from_directory(DOWNLOAD_DIRECTORY, PATH, as_attachment=True)
        except NotFound:
            ns.abort(404, 'Arquivo não gerado (Algum erro ocorreu no processo de treinamento)')


if __name__ == '__main__':
    app.run(debug=True)
