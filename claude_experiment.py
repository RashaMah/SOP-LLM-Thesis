import os
import csv
import time
import requests
from datetime import datetime

# ============================================================
# 1. OPENROUTER API KEY
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set in the PyCharm run configuration."
    )

API_KEY = API_KEY.strip()

# ============================================================
# 2. FROZEN EXPERIMENT CONFIGURATION
# ============================================================

MODEL = "anthropic/claude-sonnet-5"
TEMPERATURE = 0
MAX_TOKENS = 4000

URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# 3. FROZEN SOP GENERATION PROMPT
# ============================================================

BASE_PROMPT = """
You are tasked with generating a professional Standard Operating Procedure (SOP) for a logistics or warehouse process based exclusively on the process information provided below.

Your goal is to transform the supplied process scenario into a clear, complete, logically structured, and operationally usable SOP.

Important Instructions:

1. Use only the information provided in the process scenario.
2. Do not invent company-specific systems, departments, software, legal requirements, standards, numerical thresholds, approval times, document names, roles, technologies, or operational rules unless they are explicitly stated in the scenario.
3. Where the scenario does not provide sufficient information for a specific operational detail, do not guess. Instead, state the limitation clearly where necessary or formulate the procedure without introducing unsupported assumptions.
4. Preserve all relevant process constraints, decision conditions, exceptions, and dependencies contained in the scenario.
5. Ensure that activities appear in a logically correct operational sequence.
6. Clearly distinguish responsibilities between the roles provided in the scenario.
7. Explicitly address exception or deviation handling where the scenario contains an abnormal condition.
8. Do not remove important information from the source scenario merely to make the SOP shorter.
9. The SOP must be understandable to a warehouse or logistics employee who has not seen the original scenario.
10. Do not describe or discuss how you generated the SOP. Output only the SOP itself.

Required SOP Structure:

1. SOP Title
2. Purpose
3. Scope
4. Roles and Responsibilities
5. Required Inputs / Information
6. Expected Outputs
7. Procedure
8. Exception and Deviation Handling
9. Process Completion Criteria
"""

# ============================================================
# 4. FINAL 12 SCENARIOS
# ============================================================

scenarios = {

    "GR-A": """
Process: Goods Receipt / Receiving

Scenario: Standard supplier delivery to a warehouse.

Objective:
Receive an expected shipment correctly, verify it against the available documentation, record the receipt and transfer accepted goods into warehouse stock.

Operational information:
A scheduled supplier delivery arrives at the warehouse. The shipment has an approved purchase-order reference, shipping documentation and a packing list. Warehouse personnel must verify the documentation and compare the delivered items and quantities with the expected shipment. The goods must be unloaded and visually inspected before they are accepted into stock. Accepted goods must be documented through the appropriate receiving record and subsequently moved to the designated storage area. Warehouse stock records must be updated after receipt.

Roles involved:
Warehouse Manager/Store Manager, Storekeeper/Warehouse Staff, receiving or inspection personnel.

Constraints / decision conditions:
The warehouse may only accept goods associated with valid shipment documentation. Product identity and received quantity must correspond to the expected shipment. Goods should not be entered into normal stock before the required receiving checks have been completed.

Expected outcome:
The shipment is verified, accepted, documented, recorded in inventory and transferred to the appropriate storage location.
""",

    "GR-B": """
Process: Goods Receipt / Receiving

Scenario: Incoming shipment contains damaged goods and a quantity discrepancy.

Objective:
Process an incoming shipment while correctly handling discrepancies identified during receiving.

Operational information:
A supplier delivery arrives with valid shipping documentation. During unloading and inspection, warehouse staff identify two problems: part of the shipment is visibly damaged and the physical quantity of one item does not match the quantity indicated in the documentation. The unaffected items are otherwise suitable for receipt. The discrepancy and damage must be documented and communicated to the responsible logistics/supply function. The affected items must not be treated as normal usable stock until a decision has been made.

Roles involved:
Warehouse Manager/Store Manager, Storekeeper/Warehouse Staff, inspection personnel, responsible Supply/Logistics function.

Constraints / decision conditions:
The physical delivery must be compared with the order and packing documentation. Any mismatch must be reported. Damaged or questionable goods must be clearly identified and separated from normal available stock. Inventory records must reflect only the accepted quantities/status.

Expected outcome:
Accepted goods are correctly received; damaged/discrepant goods are controlled and documented; the discrepancy is escalated; and inventory records accurately reflect the result.
""",

    "RL-A": """
Process: Reverse Logistics and Damaged Goods Handling

Scenario:
Goods already held in the warehouse are identified as non-conforming and must be returned.

Objective:
Remove identified non-conforming goods from usable inventory and prepare them for controlled return.

Operational information:
A stored batch is identified as unsuitable for normal use because of a documented quality problem. Warehouse personnel must identify the affected quantity, remove it from normal usable stock, place it in a clearly separated holding area and update the relevant stock records. The responsible logistics/supply function must coordinate the further return or disposition process. Required supporting documentation must accompany the goods when they leave the warehouse.

Roles involved:
Supply Officer/Logistics function, Warehouse or Store Manager, Storekeeper, warehouse staff.

Constraints / decision conditions:
Affected goods must not remain available for normal picking. Their identity and quantity must remain traceable. The goods may only leave the warehouse through an authorized and documented process.

Expected outcome:
Affected stock is isolated, correctly recorded and prepared for authorized return without contaminating normal available inventory.
""",

    "RL-B": """
Process: Reverse Logistics and Damaged Goods Handling

Scenario:
A formal recall notice is received for a product currently stored in the warehouse.

Objective:
Identify, isolate and return all warehouse stock affected by a product recall.

Operational information:
The Supply function receives a recall notification concerning a product because of a quality or safety concern. Several quantities of the affected product may be located in warehouse stock. Warehouse personnel must identify the affected stock, stop it from being issued, segregate it in a separate area and mark it appropriately. Stock records must be updated. Relevant offices and external parties must be coordinated with before the goods are dispatched for recall.

Roles involved:
Supply Officer, Store Manager, Store Supervisor, Storekeeper, Store Helper/Warehouse Staff, Security.

Constraints / decision conditions:
Affected stock must be prevented from normal release immediately after the recall is confirmed. All quantities moved out of usable stock must remain traceable. Dispatch requires appropriate documentation and security/release controls.

Expected outcome:
All affected quantities are identified, blocked from use, segregated, recorded and released through a controlled reverse-logistics process.
""",

    "IR-A": """
Process: Inventory Replenishment and Stock Realignment

Scenario:
Picking stock has fallen to a low level while reserve stock is available.

Objective:
Replenish the active picking location from warehouse reserve stock before it becomes insufficient for expected picking activity.

Operational information:
The warehouse uses separate active picking locations and reserve storage. During stock review, warehouse personnel determine that the quantity of an item in its picking location has become low. Additional quantity of the same item is available in reserve storage. The required stock must be transferred from reserve storage to the designated picking location and the inventory/location records must remain accurate after the movement.

Roles involved:
Storekeeper/Warehouse Staff, Warehouse or Store Manager.

Constraints / decision conditions:
The correct product and stock batch must be used. The movement must not create an unexplained difference between physical and recorded inventory. Any applicable stock-rotation rule must continue to be respected.

Expected outcome:
The picking location contains sufficient stock for continued operations and both physical and recorded quantities remain aligned.
""",

    "IR-B": """
Process: Inventory Replenishment and Stock Realignment

Scenario:
Current stock cannot satisfy all known future demand.

Objective:
Detect an inventory shortage and initiate an appropriate replenishment or stock-realignment response without incorrectly promising unavailable stock.

Operational information:
Inventory review shows that the quantity currently available for an item is lower than the quantity required for planned releases. Additional demand is already known. Warehouse personnel must verify the available stock and communicate the shortage to the responsible supply/planning function. Existing stock and planned releases must be considered when determining the next action. If stock from another location can be realigned, the transfer must be coordinated; otherwise replenishment must be initiated through the appropriate supply process.

Roles involved:
Inventory/Supply Officer, Warehouse Manager, Storekeeper, relevant planning or logistics function.

Constraints / decision conditions:
The procedure must not assume stock exists when it does not. Available stock must be verified before decisions are made. Pending requirements should not be silently cancelled or fulfilled with invented quantities.

Expected outcome:
The shortage is identified and documented, stakeholders are informed, and either replenishment or authorized stock realignment is initiated.
""",

    "OP-A": """
Process: Order Picking

Scenario:
A warehouse operator must prepare a normal multi-item picking wave.

Objective:
Pick all required products and quantities accurately from their assigned warehouse locations and transfer the completed pick for the next fulfilment stage.

Operational information:
An approved picking instruction contains multiple product references, required quantities and corresponding storage locations. One operator or picking team is assigned to the picking task. The warehouse uses identified storage locations and the order may contain items from several locations. Items must be picked according to the authorized picking information and transferred to the designated staging/packing area after completion.

Roles involved:
Storekeeper/Picker, Warehouse Supervisor or Manager.

Constraints / decision conditions:
Only the product references and quantities stated in the picking instruction may be picked. Each picked line must be checked against the instruction. Inventory/location records must remain accurate after the physical movement.

Expected outcome:
All available requested items are picked accurately, checked and transferred to the next order-fulfilment stage.
""",

    "OP-B": """
Process: Order Picking

Scenario:
The picking instruction requests an item that is not physically available at its expected picking location.

Objective:
Continue the picking operation correctly without substituting or inventing unavailable stock.

Operational information:
During a multi-item picking operation, the picker reaches the listed location for one requested item but the required quantity is not physically available there. Other items in the picking instruction are available. The discrepancy must be recorded and communicated to the responsible warehouse personnel so that stock records can be checked and reconciled. Available items may continue through the appropriate process, but the missing quantity must not be falsely marked as picked.

Roles involved:
Picker/Storekeeper, Storekeeper responsible for stock records, Warehouse Supervisor/Manager.

Constraints / decision conditions:
The picker must follow the authorized picking instruction. An alternative batch or quantity must not be selected unless permitted by the warehouse stock rules. The discrepancy between the physical location and the inventory/picking information must be recorded.

Expected outcome:
Available items are picked correctly; unavailable quantities are not falsely confirmed; and the discrepancy is escalated for stock reconciliation or back-order handling.
""",

    "ST-A": """
Process: Stock Transfer and Redistribution

Scenario:
Approved stock is transferred from one warehouse to another warehouse within the same logistics network.

Objective:
Prepare, release and document an authorized stock transfer while preserving inventory traceability.

Operational information:
An approved transfer request identifies the goods and quantities to be moved from the sending warehouse to another warehouse. The required goods are available. Warehouse personnel must reserve, pick and stage the correct goods and prepare the required release and transport documentation. The outgoing quantity must be removed from the sending warehouse records. The receiving warehouse will subsequently confirm receipt through its receiving process.

Roles involved:
Supply/Inventory Officer, Warehouse Manager, Storekeeper/Warehouse Staff, receiving warehouse personnel.

Constraints / decision conditions:
Stock may only be released against an approved transfer request. Requested items must remain clearly associated with the correct transfer during staging and loading. Inventory records must reflect the outgoing movement.

Expected outcome:
The correct goods are released to the destination warehouse with complete documentation and traceable stock records.
""",

    "ST-B": """
Process: Stock Transfer and Redistribution

Scenario:
One location holds surplus stock approaching a risk of expiration while another location faces insufficient stock.

Objective:
Realign inventory between locations to reduce the risk of expiry and prevent stockout.

Operational information:
Stock analysis identifies excess quantity of an item at one facility and insufficient quantity at another facility. The responsible Supply function determines that redistribution is required. The source location must prepare the agreed quantity for collection or dispatch. The receiving location must receive and record the stock. Picking, packing, transport handover, receiving and stock-record updates must remain traceable.

Roles involved:
Supply Officer, Store Manager, Store Supervisor, Storekeeper, Store Helper/Warehouse Staff, transport/3PL personnel, receiving facility.

Constraints / decision conditions:
The redistribution quantity must be based on stock information and an authorized decision. Stock must remain traceable throughout the transfer. The movement should not create an unmanaged shortage at the source location.

Expected outcome:
Inventory is redistributed from the surplus location to the shortage location and both locations' stock records correctly reflect the movement.
""",

    "PD-A": """
Process: Packing and Dispatch Preparation

Scenario:
A correctly picked order is ready for packing and shipment.

Objective:
Pack, document, stage and release the prepared goods accurately for transport.

Operational information:
A completed picking order has been moved to the staging area. The items must be packed appropriately, associated with the correct order and prepared for dispatch. A packing list and required transport/release documentation must be prepared. Before release, the shipment must be checked against the dispatch documentation. Stock records must reflect the released goods.

Roles involved:
Warehouse Staff/Storekeeper, Warehouse or Store Manager, loading/checking personnel, transport representative.

Constraints / decision conditions:
Orders in the staging area must remain clearly separated. Packed goods must correspond to the authorized order. The outgoing quantity must be verified before custody is transferred to the carrier or recipient.

Expected outcome:
The correct shipment is packed, documented, verified and handed over for transport with accurate stock records.
""",

    "PD-B": """
Process: Packing and Dispatch Preparation

Scenario:
A large outgoing shipment requires coordinated staging and loading.

Objective:
Prepare and dispatch a large shipment without mixing orders or exceeding loading/operational constraints.

Operational information:
A large approved shipment has been picked. The volume of goods requires a planned loading operation rather than immediate release. Warehouse personnel must organize the staging area, keep the shipment clearly identified, prepare packing and dispatch documentation and coordinate loading. The vehicle load must correspond to the authorized shipment and documentation. Physical warehouse limitations and vehicle weight/volume limitations must be considered.

Roles involved:
Warehouse Manager, Storekeeper/Warehouse Staff, tally/checking or security personnel, transport personnel.

Constraints / decision conditions:
Goods belonging to different orders must not be mixed. Loading must respect vehicle and operational capacity constraints. Quantities loaded must be checked against the dispatch/Waybill information before release.

Expected outcome:
The complete authorized shipment is staged, packed, loaded, verified and released without order mixing or undocumented quantity differences.
"""
}

# ============================================================
# 5. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = "CLAUDE_OFFICIAL_OUTPUTS"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

log_rows = []

# ============================================================
# 6. RUN ALL 12 SCENARIOS
# ============================================================

for scenario_id, scenario_text in scenarios.items():

    print("=" * 60)
    print(f"Running {scenario_id}...")
    print("=" * 60)

    full_input = (
        BASE_PROMPT.strip()
        + "\n\nPROCESS SCENARIO:\n\n"
        + scenario_text.strip()
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": full_input
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,

        # Pin the same provider used in the successful pilot.
        "provider": {
            "only": ["amazon-bedrock"],
            "allow_fallbacks": False
        }
    }

    start_time = datetime.now().isoformat(timespec="seconds")

    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            json=payload,
            timeout=180
        )

        end_time = datetime.now().isoformat(timespec="seconds")

        if response.status_code != 200:
            print(f"ERROR: {scenario_id}")
            print("HTTP status:", response.status_code)
            print(response.text)

            log_rows.append({
                "scenario_id": scenario_id,
                "model": MODEL,
                "provider": "",
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "status": "ERROR",
                "http_status": response.status_code,
                "start_time": start_time,
                "end_time": end_time,
                "output_file": ""
            })
            break

        data = response.json()

        output_text = data["choices"][0]["message"]["content"].strip()
        provider = data.get("provider", "")

        if not output_text:
            print(f"ERROR: {scenario_id}")
            print("HTTP 200 received, but output was empty.")
            break

        output_filename = f"CLAUDE_{scenario_id}_RAW.txt"
        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(output_text)

        print(f"SUCCESS: {scenario_id}")
        print("Model:", data.get("model", MODEL))
        print("Provider:", provider)
        print(f"Saved as: {output_filename}")

        log_rows.append({
            "scenario_id": scenario_id,
            "model": data.get("model", MODEL),
            "provider": provider,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "status": "SUCCESS",
            "http_status": response.status_code,
            "start_time": start_time,
            "end_time": end_time,
            "output_file": output_filename
        })

    except requests.RequestException as exc:

        end_time = datetime.now().isoformat(timespec="seconds")

        print(f"ERROR: {scenario_id}")
        print(str(exc))

        log_rows.append({
            "scenario_id": scenario_id,
            "model": MODEL,
            "provider": "",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "status": "REQUEST_ERROR",
            "http_status": "",
            "start_time": start_time,
            "end_time": end_time,
            "output_file": ""
        })
        break

    time.sleep(2)

# ============================================================
# 7. SAVE EXPERIMENT LOG
# ============================================================

log_path = os.path.join(
    OUTPUT_FOLDER,
    "CLAUDE_EXPERIMENT_LOG.csv"
)

fieldnames = [
    "scenario_id",
    "model",
    "provider",
    "temperature",
    "max_tokens",
    "status",
    "http_status",
    "start_time",
    "end_time",
    "output_file"
]

with open(log_path, "w", newline="", encoding="utf-8") as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(log_rows)

successful_outputs = sum(
    1 for row in log_rows
    if row["status"] == "SUCCESS"
)

print()
print("=" * 60)
print("BATCH FINISHED")
print("=" * 60)
print(f"Successful outputs: {successful_outputs} / {len(scenarios)}")
print(f"Files saved in: {OUTPUT_FOLDER}")