from flask import Flask, render_template
from markdown import markdown
import os
import env


app = Flask(__name__)


@app.route("/")
def index():
    rendered_list = list()
    for re in os.listdir('./reports/'):
        with open(f'./reports/{re}', encoding='utf-8') as r:
            rendered_list.append({"html": markdown(
                r.read()[:100]) + '...', "date": re.split('_')[1].split('.')[0]})
    return render_template("index.html", contents=rendered_list)


@app.route("/reports/<args>")
def value(args):
    rendered_list = list()
    with open(f'./reports/report_{args}.md', encoding='utf-8') as r:
        rendered_list = {"html": markdown(r.read()), "date": args}
    return render_template("detail.html", content=rendered_list)


if __name__ == "__main__":
    app.run("0.0.0.0", port=env.PORT)
