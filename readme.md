# first run these export command with proper credentials
export EMAIL_SENDER="your-gmail@gmail.com"
export EMAIL_PASSWORD="your-16-character-app-password"
export EMAIL_RECIPIENT="recipient@email.com"

# command : python3 log_watcher.py training.log --custom-metrics custom_metrics.json
# custom_metrics.json may look like below 
{
    "CustomLoss": "custom_loss[:\\s]+([\\d\\.]+)",
    "CustomMetric": "metric[:\\s]+([\\d\\.]+)"
}