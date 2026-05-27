using UnityEngine;
using UnityEngine.UI;

public class ProgressBarManagerBlind : MonoBehaviour
{
    public Image progressBarFillBlind; // Assign this in the Inspector
    public Image progressBarBackgroundBlind; // Assign this in the Inspector
    private float holdTime = 5.0f; // Total time to hold the button
    private float currentHoldTime = 0f;
    private bool isButtonHeld = false;

    void Awake()
    {
        progressBarFillBlind.gameObject.SetActive(false);
        progressBarBackgroundBlind.gameObject.SetActive(false);
        AlignBackgroundWithFill();
    }
    
    void Update()
    {
        if (isButtonHeld)
        {
            currentHoldTime += Time.deltaTime;
            UpdateProgressBar(currentHoldTime / holdTime);
        }
    }

    public void StartButtonHold()
    {
        isButtonHeld = true;
        currentHoldTime = 0f;
        progressBarFillBlind.gameObject.SetActive(true);
        progressBarBackgroundBlind.gameObject.SetActive(true);
    }

    public void EndButtonHold()
    {
        isButtonHeld = false;
        progressBarFillBlind.gameObject.SetActive(false);
        progressBarBackgroundBlind.gameObject.SetActive(false);
    }

    void UpdateProgressBar(float progress)
    {
        progressBarFillBlind.fillAmount = progress;
    }

    private void AlignBackgroundWithFill()
    {
        if (progressBarFillBlind != null && progressBarBackgroundBlind != null)
        {
            RectTransform fillTransform = progressBarFillBlind.GetComponent<RectTransform>();
            RectTransform backgroundTransform = progressBarBackgroundBlind.GetComponent<RectTransform>();

            backgroundTransform.position = fillTransform.position;
            backgroundTransform.sizeDelta = fillTransform.sizeDelta;
            backgroundTransform.anchorMin = fillTransform.anchorMin;
            backgroundTransform.anchorMax = fillTransform.anchorMax;
            backgroundTransform.pivot = fillTransform.pivot;

            // Ensure the background is always rendered behind the fill
            progressBarBackgroundBlind.transform.SetSiblingIndex(0);
            progressBarFillBlind.transform.SetSiblingIndex(1);
        }
    }
}
