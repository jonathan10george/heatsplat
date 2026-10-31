"""Audit report writer. Template-based today; the Claude agent slots in behind the same function."""

from __future__ import annotations

from datetime import date

GENERATOR_TEMPLATE = "template-v1"


def _leak_section(leak: dict) -> str:
    c = leak["centroid"]
    cost = leak["cost"]
    return "\n".join(
        [
            f"### {leak['id']}: {leak['likely_cause']}",
            "",
            f"- **Location:** x {c[0]:.1f} m, height {c[1]:.1f} m, z {c[2]:.1f} m",
            f"- **Warmer than surroundings by:** {leak['mean_delta_c']:.1f} °C (peak {leak['max_delta_c']:.1f} °C)",
            f"- **Affected area:** {leak['area_m2']:.2f} m²",
            f"- **Estimated loss:** {cost['watts']:.0f} W, about {cost['kwh_per_year']:.0f} kWh/yr, {cost['currency']} {cost['cost_per_year']:.2f}/yr",
            f"- **Recommended fix:** {recommended_fix(leak['likely_cause'])}",
            "",
        ]
    )


def recommended_fix(cause: str) -> str:
    fixes = {
        "Gap under a door or at the sill plate": "Fit a door sweep and threshold seal; caulk or foam the sill plate to foundation joint.",
        "Air leak at the eave / top plate (attic bypass)": "Air-seal the top plate from the attic side with foam or caulk, then top up attic insulation.",
        "Failed seal around a window or door frame": "Remove trim, low-expansion foam the rough-opening gap, re-caulk the exterior frame joint.",
        "Missing or settled wall insulation": "Dense-pack cellulose or inject foam into the cavity through small exterior holes; patch and paint.",
    }
    return fixes.get(cause, "Have an auditor confirm with a blower-door test before repairing.")


def render_report(scene_meta: dict, leaks_payload: dict) -> str:
    """Markdown audit report from the scene metadata and current leaks payload."""
    summary = leaks_payload["summary"]
    leaks = leaks_payload["leaks"]
    lines = [
        f"# Thermal envelope audit: {scene_meta.get('name', leaks_payload['scene_id'])}",
        "",
        f"*Generated {date.today().isoformat()} by Heatsplat from a {scene_meta.get('n_gaussians', 0):,}-Gaussian thermal scan.*",
        "",
        "## Summary",
        "",
        f"- **Leaks found:** {summary['n_leaks']}",
        f"- **Estimated combined loss:** {summary['total_watts']:.0f} W during heating, about {summary['currency']} {summary['total_cost_per_year']:.2f} per year",
        f"- **Detection settings:** warmer than local baseline by at least {leaks_payload['params']['min_delta_c']} °C "
        f"(z ≥ {leaks_payload['params']['min_zscore']}), clusters within {leaks_payload['params']['cluster_eps_m']} m",
        "",
        "## Prioritised findings",
        "",
    ]
    if not leaks:
        lines.append("No leaks exceeded the detection thresholds. Consider lowering the sensitivity threshold or rescanning at dusk with a larger indoor/outdoor temperature difference.")
    lines.extend(_leak_section(leak) for leak in leaks)
    lines.extend(
        [
            "## Method and caveats",
            "",
            "Each Gaussian in the 3D scan was compared with the median temperature of its surface neighbourhood; "
            "clusters that are robustly warmer were grouped into leaks. Cost figures use a simple surface-loss model "
            f"(h = {leaks_payload['cost_model']['surface_coefficient_w_m2k']} W/m²K, "
            f"{leaks_payload['cost_model']['heating_hours_per_year']:.0f} heating hours, "
            f"{summary['currency']} {leaks_payload['cost_model']['price_per_kwh']}/kWh) and are for prioritisation only. "
            "Confirm with a blower-door test before major work.",
            "",
        ]
    )
    return "\n".join(lines)
