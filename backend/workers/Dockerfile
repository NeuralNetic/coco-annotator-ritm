FROM jsbroks/coco-annotator:python-env

WORKDIR /workspace/

# Install python package dependices
COPY ./backend/ /workspace/

RUN pip install requests_toolbelt

EXPOSE 5555
EXPOSE 9009
CMD celery -A workers worker -l info
