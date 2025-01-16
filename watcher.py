import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class MLLogWatcher:
    """Base class for ML training log watching and reporting."""

    def __init__(
        self,
        log_file: str,
        check_interval: int = 300,  # 5 minutes default
        email_interval: int = 3600,  # 1 hour default
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = None,
        sender_password: str = None,
        recipient_email: str = None,
        plot_dir: str = "training_plots"
    ):
        self.log_file = log_file
        self.check_interval = check_interval
        self.email_interval = email_interval
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.plot_dir = Path(plot_dir)

        # File reading state
        self.last_position = 0
        self.buffer = []
        self.last_email_time = time.time()

        # Training tracking state
        self.training_start_time = time.time()
        self.current_epoch = 0
        self.best_metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'val_loss': float('inf'),
            'val_accuracy': 0.0
        }

    def setup_email_config(self) -> None:
        """Set up email configuration from environment variables if not provided."""
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            self.sender_email = os.getenv('EMAIL_SENDER')
            self.sender_password = os.getenv('EMAIL_PASSWORD')
            self.recipient_email = os.getenv('EMAIL_RECIPIENT')

        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            raise ValueError("Email configuration is incomplete. Please provide all email credentials.")

    def check_file_changes(self) -> Optional[str]:
        """Check for new content in the log file."""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_content = f.read()
                if new_content:
                    self.last_position = f.tell()
                    self.buffer.append(new_content)
                    return new_content
                return None
        except Exception as e:
            print(f"Error reading log file: {str(e)}")
            return None

    def should_send_email(self) -> bool:
        """Determine if it's time to send an email based on the interval."""
        return (time.time() - self.last_email_time) >= self.email_interval

    def format_email_body(self, new_content: str) -> str:
        """Format the email body with training metrics and analysis."""
        duration = time.time() - self.training_start_time
        hours = duration // 3600
        minutes = (duration % 3600) // 60

        body = f"""
        <html>
        <body>
        <h2>Training Progress Report</h2>
        <p>Training Duration: {int(hours)}h {int(minutes)}m</p>
        <p>Current Epoch: {self.current_epoch}</p>
        
        <h3>Best Metrics:</h3>
        <ul>
            <li>Best Loss: {self.best_metrics['loss']:.4f}</li>
            <li>Best Accuracy: {self.best_metrics['accuracy']:.4f}</li>
            <li>Best Validation Loss: {self.best_metrics['val_loss']:.4f}</li>
            <li>Best Validation Accuracy: {self.best_metrics['val_accuracy']:.4f}</li>
        </ul>

        <h3>Recent Training Log:</h3>
        <pre>{new_content}</pre>
        </body>
        </html>
        """
        return body

    def send_email(self, subject: str, body: str) -> bool:
        """Send email with the training progress report."""
        try:
            msg = MIMEMultipart('related')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            # Create HTML message container
            msg_alternative = MIMEMultipart('alternative')
            msg.attach(msg_alternative)

            # Attach HTML version
            html_part = MIMEText(body, 'html')
            msg_alternative.attach(html_part)

            # Attach any available plots from the plot directory
            if self.plot_dir.exists():
                for plot_file in self.plot_dir.glob('*.png'):
                    with open(plot_file, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-ID', f'<{plot_file.name}>')
                        msg.attach(img)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            print(f"Email sent successfully at {datetime.now()}")
            return True

        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

    def process_buffer(self) -> None:
        """Process the accumulated buffer content."""
        # This method can be overridden by subclasses to implement
        # specific processing logic
        pass

    def watch(self) -> None:
        """Main watching loop."""
        print(f"Starting to watch ML training logs: {self.log_file}")
        print(f"Will check every {self.check_interval} seconds")
        print(f"Will send email reports every {self.email_interval} seconds")

        # Ensure email configuration is set up
        self.setup_email_config()

        # Create plot directory if it doesn't exist
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                # Check for new content
                new_content = self.check_file_changes()

                if new_content:
                    self.process_buffer()

                # If we have buffered content and it's time to send an email
                if self.buffer and self.should_send_email():
                    email_body = self.format_email_body("".join(self.buffer))
                    subject = f"ML Training Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

                    if self.send_email(subject, email_body):
                        self.buffer = []  # Clear buffer after successful send
                        self.last_email_time = time.time()

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\nStopping log watcher...")
                # Send final report
                if self.buffer:
                    email_body = self.format_email_body("".join(self.buffer))
                    subject = f"Final Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    self.send_email(subject, email_body)
                break

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                time.sleep(self.check_interval)

    def cleanup(self) -> None:
        """Cleanup resources before shutting down."""
        # This method can be overridden by subclasses to implement
        # specific cleanup logic
        pass
