FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install poetry
RUN poetry install
EXPOSE 5000
CMD ["poetry", "run", "python", "flask_app.py"] 
