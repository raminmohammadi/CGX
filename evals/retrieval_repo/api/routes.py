"""HTTP API surface for the billing service."""

from flask import Blueprint, Flask

bp = Blueprint("billing", __name__)


@bp.route("/invoices", methods=["GET"])
def list_invoices():
    """Return every invoice for the current tenant."""
    return []


@bp.route("/invoices/<invoice_id>/refund", methods=["POST"])
def refund_invoice(invoice_id):
    """Issue a refund against a previously paid invoice."""
    return {"refunded": invoice_id}


def create_app():
    """Application factory that wires up the billing blueprint."""
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app
