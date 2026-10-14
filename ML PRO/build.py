# Injects the trained model (bundle.json) into template.html -> sales_forecast_app.html
s = open('template.html').read().replace('__BUNDLE__', open('bundle.json').read())
open('sales_forecast_app.html', 'w').write(s)
print('Built sales_forecast_app.html')
