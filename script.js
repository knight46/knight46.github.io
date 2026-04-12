const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const yearTarget = document.getElementById("year");
if (yearTarget) {
    yearTarget.textContent = new Date().getFullYear();
}

document.querySelectorAll("[data-scroll]").forEach((link) => {
    link.addEventListener("click", (event) => {
        const targetId = link.getAttribute("href");
        if (!targetId || !targetId.startsWith("#")) {
            return;
        }

        const target = document.querySelector(targetId);
        if (!target) {
            return;
        }

        event.preventDefault();
        target.scrollIntoView({
            behavior: prefersReducedMotion ? "auto" : "smooth",
            block: "start"
        });
    });
});

const revealItems = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(
        (entries, observer) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting) {
                    return;
                }

                entry.target.classList.add("is-visible");
                observer.unobserve(entry.target);
            });
        },
        {
            threshold: 0.18
        }
    );

    revealItems.forEach((item) => revealObserver.observe(item));
} else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
}

const sectionIds = ["about", "skills", "projects", "journey", "contact"];
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const sectionMap = new Map(
    sectionIds
        .map((id) => [id, document.getElementById(id)])
        .filter(([, element]) => element)
);

if ("IntersectionObserver" in window && sectionMap.size) {
    const sectionObserver = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

            if (!visible) {
                return;
            }

            const activeId = visible.target.id;
            navLinks.forEach((link) => {
                const isActive = link.getAttribute("href") === `#${activeId}`;
                link.classList.toggle("is-active", isActive);
            });
        },
        {
            rootMargin: "-35% 0px -45% 0px",
            threshold: [0.15, 0.35, 0.6]
        }
    );

    sectionMap.forEach((section) => sectionObserver.observe(section));
}

const canvas = document.getElementById("hero-canvas");
if (canvas && !prefersReducedMotion) {
    const context = canvas.getContext("2d");
    if (context) {
        const pointer = {
            x: 0,
            y: 0,
            active: false
        };

        let animationId = 0;
        let width = 0;
        let height = 0;
        let particles = [];

        const createParticle = () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.55,
            vy: (Math.random() - 0.5) * 0.55,
            size: 1 + Math.random() * 2.6
        });

        const resizeCanvas = () => {
            const ratio = window.devicePixelRatio || 1;
            width = canvas.clientWidth;
            height = canvas.clientHeight;
            canvas.width = Math.floor(width * ratio);
            canvas.height = Math.floor(height * ratio);
            context.setTransform(ratio, 0, 0, ratio, 0, 0);

            const count = Math.max(32, Math.min(88, Math.floor((width * height) / 22000)));
            particles = Array.from({ length: count }, createParticle);
        };

        const draw = () => {
            context.clearRect(0, 0, width, height);

            for (let index = 0; index < particles.length; index += 1) {
                const particle = particles[index];
                particle.x += particle.vx;
                particle.y += particle.vy;

                if (particle.x <= -20 || particle.x >= width + 20) {
                    particle.vx *= -1;
                }
                if (particle.y <= -20 || particle.y >= height + 20) {
                    particle.vy *= -1;
                }

                if (pointer.active) {
                    const dx = particle.x - pointer.x;
                    const dy = particle.y - pointer.y;
                    const distance = Math.hypot(dx, dy);
                    if (distance > 0 && distance < 160) {
                        const force = (160 - distance) / 1600;
                        particle.vx += (dx / distance) * force;
                        particle.vy += (dy / distance) * force;
                    }
                }

                particle.vx *= 0.995;
                particle.vy *= 0.995;

                const gradient = context.createRadialGradient(
                    particle.x,
                    particle.y,
                    0,
                    particle.x,
                    particle.y,
                    particle.size * 6
                );
                gradient.addColorStop(0, "rgba(255, 208, 138, 0.9)");
                gradient.addColorStop(0.45, "rgba(122, 216, 199, 0.28)");
                gradient.addColorStop(1, "rgba(122, 216, 199, 0)");

                context.fillStyle = gradient;
                context.beginPath();
                context.arc(particle.x, particle.y, particle.size * 5, 0, Math.PI * 2);
                context.fill();

                context.fillStyle = "rgba(255, 255, 255, 0.7)";
                context.beginPath();
                context.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                context.fill();
            }

            for (let index = 0; index < particles.length; index += 1) {
                for (let next = index + 1; next < particles.length; next += 1) {
                    const particle = particles[index];
                    const target = particles[next];
                    const dx = particle.x - target.x;
                    const dy = particle.y - target.y;
                    const distance = Math.hypot(dx, dy);

                    if (distance > 120) {
                        continue;
                    }

                    context.strokeStyle = `rgba(255, 255, 255, ${0.16 - distance / 900})`;
                    context.lineWidth = 1;
                    context.beginPath();
                    context.moveTo(particle.x, particle.y);
                    context.lineTo(target.x, target.y);
                    context.stroke();
                }
            }

            if (pointer.active) {
                const pointerGlow = context.createRadialGradient(
                    pointer.x,
                    pointer.y,
                    0,
                    pointer.x,
                    pointer.y,
                    180
                );
                pointerGlow.addColorStop(0, "rgba(255, 122, 89, 0.18)");
                pointerGlow.addColorStop(1, "rgba(255, 122, 89, 0)");

                context.fillStyle = pointerGlow;
                context.beginPath();
                context.arc(pointer.x, pointer.y, 180, 0, Math.PI * 2);
                context.fill();
            }

            animationId = window.requestAnimationFrame(draw);
        };

        const handlePointerMove = (event) => {
            const rect = canvas.getBoundingClientRect();
            pointer.x = event.clientX - rect.left;
            pointer.y = event.clientY - rect.top;
            pointer.active = true;
        };

        const handlePointerLeave = () => {
            pointer.active = false;
        };

        const syncVisibility = () => {
            if (document.hidden) {
                window.cancelAnimationFrame(animationId);
                return;
            }

            window.cancelAnimationFrame(animationId);
            animationId = window.requestAnimationFrame(draw);
        };

        resizeCanvas();
        draw();

        window.addEventListener("resize", resizeCanvas);
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("blur", handlePointerLeave);
        document.addEventListener("mouseleave", handlePointerLeave);
        document.addEventListener("visibilitychange", syncVisibility);
    }
}
