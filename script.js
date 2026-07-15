document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");

    if (!form) return;

    form.addEventListener("submit", function (e) {

        const tenure = document.querySelector('input[name="tenure"]');
        const monthlyCharges = document.querySelector('input[name="MonthlyCharges"]');
        const totalCharges = document.querySelector('input[name="TotalCharges"]');

        if (
            tenure.value === "" ||
            monthlyCharges.value === "" ||
            totalCharges.value === ""
        ) {
            alert("Please fill all required fields.");
            e.preventDefault();
            return;
        }

        if (Number(tenure.value) < 0 || Number(tenure.value) > 72) {
            alert("Tenure must be between 0 and 72 months.");
            e.preventDefault();
            return;
        }

        if (Number(monthlyCharges.value) < 0) {
            alert("Monthly Charges cannot be negative.");
            e.preventDefault();
            return;
        }

        if (Number(totalCharges.value) < 0) {
            alert("Total Charges cannot be negative.");
            e.preventDefault();
            return;
        }

    });

});
