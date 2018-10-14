FROM python:3.6.4

ENV FLASK_APP serverprueba.py
ENV FLASK_ENV development
ENV FLASK_DEBUG 0

WORKDIR /app

RUN mkdir -p /app/result-studio/clasificador-reacciones_49
RUN mkdir -p /app/result-studio/clasificador-conductas-individuales/positiva
RUN mkdir -p /app/result-studio/clasificador-conductas-individuales/negativa
RUN mkdir -p /app/result-studio/clasificador-conductas-individuales/pregunta
RUN mkdir -p /app/result-studio/clasificador-conductas-individuales/responde
RUN mkdir -p /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-plain_2
RUN mkdir -p /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-cnn_19
RUN mkdir -p /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-rnn_66
RUN mkdir -p /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-crnn_83
RUN mkdir -p /app/lib

ADD requirements.txt /app/
ADD embeddings.db /app/
ADD app.py /app/
ADD serverprueba.py /app/
ADD configuration.py /app/
ADD model.py /app/
ADD preprocessing.py /app/
ADD database_embeddings.py /app/
ADD corrector.py /app/
ADD result-studio/clasificador-reacciones_49/model.h5 /app/result-studio/clasificador-reacciones_49/model.h5
ADD result-studio/clasificador-conductas-individuales/positiva/model.h5 /app/result-studio/clasificador-conductas-individuales/positiva/model.h5
ADD result-studio/clasificador-conductas-individuales/negativa/model.h5 /app/result-studio/clasificador-conductas-individuales/negativa/model.h5
ADD result-studio/clasificador-conductas-individuales/pregunta/model.h5 /app/result-studio/clasificador-conductas-individuales/pregunta/model.h5
ADD result-studio/clasificador-conductas-individuales/responde/model.h5 /app/result-studio/clasificador-conductas-individuales/responde/model.h5
ADD result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-plain_2/model.h5 /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-plain_2/model.h5
ADD result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-cnn_19/model.h5 /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-cnn_19/model.h5
ADD result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-rnn_66/model.h5 /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-rnn_66/model.h5
ADD result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-crnn_83/model.h5 /app/result-studio/clasificadores-nuevos/fserrano-clasificador-reacciones-crnn_83/model.h5
ADD lib /app/lib

RUN pip install -r requirements.txt
ENTRYPOINT ["flask", "run"]
CMD ["--host=0.0.0.0", "--port=8083"]

EXPOSE 8083

#ENTRYPOINT ["python"]
#CMD ["app.py"]