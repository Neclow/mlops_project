from great_expectations.render.view import DefaultJinjaPageView


# Adapted from https://discourse.greatexpectations.io/t/a-super-simple-alternative-introduction-to-great-expectations/27/5
def generate_html(obj, renderer, output_file):
    document_model = renderer.render(obj)
    with open(output_file, "w", encoding="utf-8") as writer:
        writer.write(DefaultJinjaPageView().render(document_model))
